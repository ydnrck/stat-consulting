For the severity dataset, Gamma regression was used. As a result, we built a model that estimates the expected claim size and shows the directional influence of each variable on claim size.

For the frequency dataset, Logistic regression was used. As a result, we built a model that predicts the probability of a person having or not having a claim (p_claim).

To combine both models, we first apply a Gamma regression model to the frequency dataset. As a result, we obtain the expected claim size if a person would have a claim (e_severity) for each individual in the frequency dataset.

We then multiply p_claim and e_severity to obtain the expected cost for each person.

Finally, we apply a 25% threshold to identify unprofitable customers. As a result, we obtain a table (output) that shows who the person is (e.g., unemployed male) and the expected cost of the claim for that person.
