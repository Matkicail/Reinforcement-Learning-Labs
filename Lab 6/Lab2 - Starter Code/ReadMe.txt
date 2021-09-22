
# Group Members
Matthew: 1669326
Mikayla: 1886648
Michael-John: 1851234 

## 1.1 MC Predition Exercise Q4
Yes, the values generated from the 10000 episode run form a more jaggered graph as compared to the 500000 episode run which is a lot more smooth. This is due the fact that running for more episodes means that the algorithm had more to learn from.

## 2.2 TD Control Exercise Q4
SARSA creates a policy that travels around the cliff by walking along the grid, this is a safe path.Q-learning creates a policy that walks along the edge of the cliff, and is more dangerous since there is a chance of walking off the cliff due to the e-greedy action selection. Q-learning generates the optimal policy while SARSA generates a safer policy.
The reason for the difference in policy is due to the update of Q-value. Q-learning is off-policy, taking the best actions at every step and SARSA is on-policy, taking conservative actions at each step.

## 2.2 TD Control Exercise Q5
Both methods would produce the same policy.

## 2.2 TD Control Exercise Q6
Both methods would asymptotically converge to the optimal policy.
