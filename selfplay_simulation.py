import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


# 定义目标高斯分布（真实分布）
def target_gaussian(x, mean=5, std_dev=2):
    return norm.pdf(x, loc=mean, scale=std_dev)

# 定义模型高斯分布
def model_gaussian(x, mean, std_dev):
    return norm.pdf(x, loc=mean, scale=std_dev)


def gaussian_logpdf(x, mean, std):
    """
    计算高斯分布的对数概率密度
    x: 输入样本
    mean: 高斯分布的均值
    std: 高斯分布的标准差
    """
    return -0.5 * np.log(2 * np.pi * std ** 2) - (x - mean) ** 2 / (2 * std ** 2)

def relu(x):
    return np.maximum(0, x)

def logsigmoid(x):
    return np.log(1 / (1 + np.exp(-x)))

def combined_loss(init_mean,init_std,model_mean, model_std, target_mean, target_std, lambda_weight=0.5, reference_free=False, loss_type="hinge"):
    """
    计算SPIN LOSS
    
    model_mean, model_std: 生成模型的均值和标准差
    human_data, self_data: 输入数据
    target_mean, target_std: 目标模型的均值和标准差
    lambda_weight: 权重系数
    reference_free: 是否进行无参考训练
    loss_type: 损失类型 ("sigmoid" 或 "hinge")
    """
    # 计算logps
    policy_generated_logps = gaussian_logpdf(model_mean, model_mean, model_std)  # 模型生成的logps
    policy_real_logps = gaussian_logpdf(target_mean, model_mean, model_std)  # 真实数据的logps
    opponent_generated_logps = gaussian_logpdf(model_mean, init_mean,init_std)  # 对手生成的logps
    opponent_real_logps = gaussian_logpdf(target_mean, init_mean,init_std)  # 目标数据的logps

    # 计算logratios
    pi_logratios = policy_real_logps - policy_generated_logps
    ref_logratios = opponent_real_logps - opponent_generated_logps

    # 如果是无参考训练，设置 ref_logratios 为 0
    if reference_free:
        ref_logratios = 0

    # 计算logits
    logits = pi_logratios - ref_logratios


    # 计算损失
    beta = 0.9
    if loss_type == "sigmoid":
        #print("we are using sigmoid")
        losses = logsigmoid(beta * logits)
    elif loss_type == "hinge":
        #print("we are using hinge")
        losses = relu(1 - beta * logits)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Should be one of ['sigmoid', 'hinge']")

    # 计算奖励
    # real_rewards = beta * (policy_real_logps - opponent_real_logps).detach()
    # generated_rewards = beta * (policy_generated_logps - opponent_generated_logps).detach()

    # 打印调试信息
    # print(f"losses: {losses}")
    # print(f"policy_real_logps: {policy_real_logps}")
    # print(f"policy_generated_logps: {policy_generated_logps}")
    # print(f"opponent_real_logps: {opponent_real_logps}")
    # print(f"opponent_generated_logps: {opponent_generated_logps}")
    # print(f"logits: {logits}")
    # print(f"real_rewards: {real_rewards}")
    # print(f"generated_rewards: {generated_rewards}")

    return losses



# 参数更新：使用数值梯度下降
def update_parameters(init_mean,init_std,mean, std, target_mean,target_std, learning_rate, lambda_weight):
    """
    使用数值梯度计算并更新模型的均值和标准差。
    """
    # 计算均值的梯度
    grad_mean = (combined_loss(init_mean,init_std,mean + 1e-5,std,target_mean, target_std, lambda_weight) -
                 combined_loss(init_mean,init_std,mean, std,target_mean, target_std, lambda_weight)) / 1e-5

    # 计算标准差的梯度
    grad_std = (combined_loss(init_mean,init_std,mean, std + 1e-5,target_mean, target_std, lambda_weight) -
                combined_loss(init_mean,init_std,mean, std,target_mean, target_std, lambda_weight)) / 1e-5

    # 更新参数
    mean -= learning_rate * grad_mean
    std -= learning_rate * grad_std
    return mean, std

# 主函数：SPIN 框架仿真
def spin_fine_tuning(target_mean=5, target_std=2, iterations=10000, human_data_size=100, self_data_size=100, learning_rate=0.001, lambda_weight=0.5):
    """
    SPIN 框架仿真，优化模型参数以逼近目标分布。
    """
    # 初始化模型的均值和标准差
    model_mean = np.random.uniform(5.5, 6.5)
    model_std = np.random.uniform(1, 3)
    init_mean=model_mean
    init_std=model_std
    print(f"初始模型参数: mean={model_mean:.4f}, std={model_std:.4f}")

    # 从目标分布生成“人类标注”数据
    # human_data = np.random.normal(target_mean, target_std, size=human_data_size)

    # 开始迭代优化
    for i in range(iterations):
        # 从当前模型分布生成“自生成”数据
        #self_data = np.random.normal(model_mean, model_std, size=self_data_size)

        # 更新模型参数
        model_mean, model_std = update_parameters(init_mean,init_std,model_mean, model_std, target_mean, target_std, learning_rate, lambda_weight)

        # print(f"type of model_mean: {type(model_mean)}")
        # print(f"type of model_std: {type(model_std)}")
        # print("model mean shape",model_mean.shape)
        # print("model_std shape",model_std.shape)


        # 每 10 次迭代输出一次进度
        if (i + 1) % 10 == 0:
            print(f"Iteration {i + 1}: mean={model_mean:.4f}, std={model_std:.4f}")

    model_mean2=model_mean
    model_std2=model_std

    # 开始迭代优化
    for i in range(iterations):
        # 从当前模型分布生成“自生成”数据
        #self_data = np.random.normal(model_mean, model_std, size=self_data_size)

        # 更新模型参数
        model_mean2, model_std2 = update_parameters(model_mean, model_std,model_mean2, model_std2, target_mean, target_std, learning_rate, lambda_weight)

        # print(f"type of model_mean: {type(model_mean)}")
        # print(f"type of model_std: {type(model_std)}")
        # print("model mean shape",model_mean.shape)
        # print("model_std shape",model_std.shape)


        # 每 10 次迭代输出一次进度
        if (i + 1) % 10 == 0:
            print(f"Iteration {i + 1}: mean={model_mean2:.4f}, std={model_std2:.4f}")


    # 可视化结果
    x = np.linspace(-10, 20, 500)
    plt.figure(figsize=(8, 6))
    plt.plot(x, target_gaussian(x, target_mean, target_std), label="Target Gaussian Function", color="red", linestyle="--")
    plt.plot(x, model_gaussian(x, init_mean, init_std), label="Model Iter-0 Gaussian Function", color="green", linewidth=2)
    plt.plot(x, model_gaussian(x, model_mean, model_std), label="Model Iter-1 Gaussian Function", color="blue", linewidth=2)
    plt.plot(x, model_gaussian(x, model_mean2, model_std2), label="Model Iter-2 Gaussian Function", color="black", linewidth=2)
    plt.title("SPIN Framework: Simulation")
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid()
    plt.show()

# 运行 SPIN 仿真
if __name__ == "__main__":
    spin_fine_tuning()
