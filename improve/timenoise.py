import torch

# sample from a logit-normal distribution
def logit_normal_sampler(m, s=1, beta_m=15, sample_num=1000000):
    y_samples = torch.randn(sample_num).reshape([m.shape[0], 1, 1, 1, 1]) * s + m
    x_samples = beta_m * (torch.exp(y_samples) / (1 + torch.exp(y_samples)))
    return x_samples

# the $\mu(t)$ function
def mu_t(t, a=5, mu_max=1):
    t = t.to('cpu')
    return 2 * mu_max * t**a - mu_max
    
# get $\beta_s$ for TimeNoise
def get_beta_s(t, a, beta_m):
    mu = mu_t(t, a=a)
    sigma_s = logit_normal_sampler(m=mu, sample_num=t.shape[0], beta_m=beta_m)
    return sigma_s

# sigma shape: [B,1,1,1,1]
bsz = 8
t = torch.randint(1000, size=(bsz, 1, 1, 1, 1))

output = get_beta_s(t/700, 5, 15)
print(output) # [B,1,1,1,1]