# Check your own code against the library

```python
import numpy as np
import screws

def my_exp6(se3mat): ...                          # your implementation

rng = np.random.default_rng(0)
cases = [screws.vec_to_se3(rng.normal(size=6)) for _ in range(20)]
screws.testing.check(my_exp6, screws.exp6, cases)  # raises on the first disagreement
```
