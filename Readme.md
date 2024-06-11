# Triangulation-Generator

In this package we use a deep Q-learning reinforcement learning model to generate fine regular star triangulations of reflexive polytopes. These triangulations provide resolutions to non-terminal singularities in the ambient toric Fano variety and Calabi-Yau hypersurfaces constructed from the polytope. 

One can also look for added geometric information, such as fibration structures in the Calabi-Yau hypersurfaces and holomorphic vector bundles that satisfy the anomaly cancellation and slope stability conditions in E8 heterotic string compactification.

For more information, please see the arxiv preprint - arXiv:2405.21017. 

## Dependencies
* This package uses functions from CYtools (https://cy.tools/). Please first instal this package before attempting to run scripts in this repository. 


## Usage
```python
# Initialize Environment
p = Polytope([
    [ 1, 0, 0, 0],
    [ 0, 1, 0, 0],
    [ 0, 0, 1, 0],
    [ 0, 0, 0, 1],
    [-1,-1, 0, 0],
    [-1,-1,-1,-1]])
s_env = SubpolytopeEnvironment(p, 2)

# Initialize agent
model = tfk.Sequential([
    Input((s_env.random_state().shape[0],)),
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(s_env.num_actions, activation='linear')
])

optim = tfk.optimizers.Adam(learning_rate = 1e-3)
model.compile(
    loss = tfk.losses.MeanSquaredError(),
    optimizer = optim,
    metrics = [tfk.metrics.MeanAbsoluteError()])

agent = Agent(model)

# Train the agent
train_agent(s_env, agent, num_epochs = 2048, verbosity = 1)
```
