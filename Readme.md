# Triangulation-Generator

In this package we use a deep Q-learning reinforcement learning model to generate fine regular star triangulations of reflexive polytopes. These triangulations provide resolutions to non-terminal singularities in the ambient toric Fano variety and Calabi-Yau hypersurfaces constructed from the polytope.

One can also look for added geometric information, such as fibration structures in the Calabi-Yau hypersurfaces and holomorphic vector bundles that satisfy the anomaly cancellation and slope stability conditions in $E_8$ heterotic string compactification.

For more information, please see the arxiv preprint - [arXiv:2405.21017](https://arxiv.org/abs/2405.21017).

## Dependencies
* This package uses functions from CYtools (https://cy.tools/). Please first install this package before attempting to run scripts in this repository.


## Quickstart
Currently supported environments are:
|Environment | Description| Arguments |
| --- | --- | --- |
| ```TriangulationEnvironment(```<br>&ensp;```polytope)```| Uses two-face encoding for generating triangulations. | - **polytope**: Polytope. |
| ```HTriangulationEnvironment(```<br>&ensp;```polytope)``` | Uses height encoding for generating triangulations. | - **polytope**: Polytope. |
| ```SubpolytopeEnvironment(```<br>&ensp;```polytope, fibration_dim)``` | Uses subspace encoding to generate subpolytope. | - **polytope**: Polytope.<br>- **fibration_dim**: Dimension of subpolytope. |

An example code for generating subpolytopes is shown below:
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


## Citation
To cite our paper:
```bibtex
@article{Berglund:2024reu,
    author = "Berglund, Per and Butbaia, Giorgi and He, Yang-Hui and Heyes, Elli and Hirst, Edward and Jejjala, Vishnu",
    title = "{Generating Triangulations and Fibrations with Reinforcement Learning}",
    eprint = "2405.21017",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    reportNumber = "QMUL-PH-24-10",
    month = "5",
    year = "2024"
}
```