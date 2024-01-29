# Meta-RL

## Second-order optimization problem

- .grad of a non-leaft tensor is being accessed
- torch.autograd.grad
- ~~Rewrite buffer to sample only a task~~
- Reduce num task episodes to 1
- Use ANIL to train the meta-policy
- Freeze the feature extractor of the meta-policy after convergence
- Use BERT structure

### Main (MAML)

- Create 2 buffers:
  - train_buffer = {$D_i$} where each $D_i$ is a buffer of trajectories sampled using the meta-policy $f_\theta$ for a single task $i$. Each $D_i$ is used to achieve the adapted policy $f_{\theta_i'}$ for task $i$.

  - val_buffer = {$D_i'$} where each $D_i'$ is a buffer of trajectories sampled using the adapted policy $f_{\theta_i'}$ for a single task $i$.

- It is easy to achieve the adapted policy $f_{\theta_i'}$ because it is a first-order optimization problem.
  - $\theta_i' = \theta - \alpha \nabla_\theta L_{D_i}(f_\theta)$
  - The implementation first clones the meta-policy parameters $\theta$ into $\theta_i'$ and then updates $\theta_i'$ using the equation $\theta_i' = \theta_i' - \alpha \nabla_{\theta_i'} L_{D_i}(f_{\theta_i'})$.

- The meta-policy is updated using the equation $\theta = \theta - \beta \nabla_\theta \sum_i L_{D_i'}(f_{\theta_i'})$
- For each task $i$, I should clone the meta-policy, do an inner loop(while retaining graph) on $D_i$, then do another inner_loop on $D_i'$. Now I have $g_i$.

### First-Order MAML(FOMAML)

### Reptile

## Graph2Seq Idea

- Initialize forward $h_f^0$ and backward $h_b^0$ embedding of each node with its embedding
- For K iterations update each node embeddings:
  - Aggregate node's forward neighbors $h_f^{k-1}$ to get $agg_f^k$
  - Pass through fully connected layer $W^k$ the concatenation of  $h_f^k$ and $agg_f^k$
  - Aggregate node's backward neighbors $h_b^{k-1}$ to get $agg_b^k$
  - Pass through fully connected layer $W^k$ the concatenation of  $h_b^k$ and $agg_b^k$
- Concatenane $h_f^K$ and $h_b^K$ as the node's new embedding
- Pass the node embeddings to another fully connected layer and apply element-wise max pooling to get the graph embedding
- Pass the graph embedding as decoder's first hidden state
- Calculate attention scores of node embeddings and decoder's hidden state and concatenate the context vector and hidden state in each decoding step.

## Inner Loop fine-tuning

- Instead of running the whole meta optimization process, I decided to find optimum hyperparameters through running a grid search over selected hyperparameters for one task. The selected task was `offload_n\offload_20` since it has DAGs with different configurations for task number $n=20$.
- For `custom` graph embedding method, `k=3 > k=1 > k=2`.
- The best number of minibatches is `16` or `8`.
- `gat` and `custom` yield almost similar results.

- Check different architectures for Q and Pi networks