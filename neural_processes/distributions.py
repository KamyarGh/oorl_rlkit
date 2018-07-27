import torch

def local_repeat(tensor, n):
    return torch.reshape(
        tensor.unsqueeze(1).expand(-1,n,-1),
        (-1, tensor.size(1))
    )

def sample_diag_gaussians(means, diags, n_samples):
    batch_size, dim = means.size(0), means.size(1)
    means = local_repeat(means, n_samples)
    diags = local_repeat(diags, n_samples)
    
    eps = torch.randn(means.size())
    samples = eps*diags + means

    print(type(batch_size), type(n_samples), type(dim))
    print((batch_size,n_samples,dim))

    return samples.reshape((batch_size,n_samples,dim))

if __name__ == '__main__':
    means = torch.randn(3,5)
    diags = torch.rand(3,5) * 0.02
    samples = sample_diag_gaussians(means, diags, 2)

    print(means)
    print(diags)
    print(samples)
