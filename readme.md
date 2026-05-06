Order to run:
python .\Pravaayu_TransformerScript.py
python .\Pruning.py
python .\Apriori.py

Support:
How frequently a rule appears in the dataset
Support(A → B) = P(A ∩ B)

Confidence:
Probability of B occurring given A
Confidence(A → B) = P(B | A)

Lift:
How much more likely A and B occur together compared to chance
Lift(A → B) = P(A ∩ B) / (P(A) \* P(B))


Support chosen >=0.12
Why: without it you get rules like something occurs in 2% of data but has 100% confidence. That’s overfitting.
This just means that this just appears in more than 12% of the patients, which is not too rare but still allows discovery.
If you go for lower support then you might get rare patterns but the data will have noise too.
If you go for highrer support then you might find only common patterns but it is more stable.

Confidence chosen is 0.75
Means we can say that if A happens then there is a 75% chance that B happens.
This is strong enought to trust but not overly strict.
Lower confidence would mean more rules but they would be weak signals.
Higher confidence would mean fewer rules but stronger signals

Note: We have also removed rules with confidence = 1. This is because this can mean that the subset is 

Lift is chosen to be more than 1.5 but less than 5
It just means that A and B occur together more than chance  also we dont go higher than 5 because instead of insight we would just be describing some patient which removes overfit rules and keeps generalizable patterns.

we set a limit on the number of conditions because too many conditions would mean that for only this exact combination of features we get this result.
