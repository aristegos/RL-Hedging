# PLEASE SEE:
Let's edit the proposal tomorrow. I did a brief initial write up (see [pdf](./COS_435_Intro_RL_Project_Proposal_Spring_2025/COS_435_Proposal_Draft_1.pdf) and see [raw](./COS_435_Intro_RL_Project_Proposal_Spring_2025/project-template.tex)). Let's take a look later on Monday and edit it. I want to add some more stuff on which RL algorithms we will use to learn optimal policy (Tabular Q-learning, Deep Q-learning, PPO, GRPO) and how we expect our algorithm to perform relative to typical delta hedging but will need to cut down on some of the words.


# GRPO_Finance
## Resources
For the papers, the general idea would be:
1. Halperin (2017) to get general overview of optimal hedging problem, the math behind it and reconciliation with analytical BS.
2. Stoiljkovic (2023) for application of Halperin model and also general discussion of performance of RL model.
3. Cao (2021) gives the most interesting model imo given with transaction costs and stochastic vol, BS is clearly suboptimal $\rightarrow$ necessitates the need for a RL model. In particular, Halperin model will yield results that try to replicate BS. Cao model creates an environment where we can do better than BS using RL.

*Ideally I'd like us to also find a paper which includes market impact of hedge

| Papers | Link | Description | Specs |
| --- | --- | --- | --- |
| Halperin QLBS (2017) | [Link](https://arxiv.org/abs/1712.04609) | Outlines main model for optimal hedging problem. Aim to minimize cost of portfolio and riskiness. Only theory, no actual implementation. | Fixed vol, Q-learning, No trans costs, discrete & continuous state-action-space applications |
| Stoiljkovic Application & Lit Review (2023) | [Link](https://arxiv.org/abs/2310.04336) | Overviews the Halperin model and then discusses various extensions in the literature. Recreates Halperin's results and compares to BS. Then applies to stochastic vol dynamics, transaction costs (same model as Cao (2021)) and various other applications (that aren't that useful to us). Benchmarks each relative to analytical models (we should do this). | Fixed & stochastic vol, Q-learning, trans costs, continuous state-action-space applications (can also be done discretely) |
| Cao trans cost extension (2021) | [Link](https://arxiv.org/abs/2103.16409) | Follows similar objective to Halperin model but now with transaction costs and also stochastic vol. I think we should use this problem formulation for our project. | Fixed & stochastic vol, DPG (policy iteration similar to REINFORCE), Trans costs, continuous state-action-space application |

## Misc Resources
| Link | Description |
| --- | --- |
| [Instructions](./Resources/Misc/COS435_Final_Project.pdf) | Final project instructions |
| [DeepSeek GRPO (2024)](https://arxiv.org/abs/2402.03300) | DeepSeek GRPO paper. Ignore everything except for the RL part (section 4) that talks about PPO vs GRPO. Use to get an idea of what GRPO is. |

