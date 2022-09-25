from dataclasses import dataclass , field
from typing import Callable , List
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class MaxEnt:

    payoffs : np.ndarray = field(default_factory = lambda : np.arange(1,7,1))
    multiplier_: float = field(init = False , default = None , repr = False)

    def predict_proba(self) -> np.ndarray:
        try:
            return self._gibbs_distr(self.multiplier_)
        except Exception as e:
            raise Exception("Model is not fitted!")

    def _gibbs_distr(self , mu) -> np.ndarray:
        partition_func_eval = self._partition_function(mu)
        return np.e**( - self.payoffs * mu)/partition_func_eval

    def _partition_function(self,mu) -> float:
        return (np.e**(  - self.payoffs * mu)).sum()

    def predict(self) -> int:
        return self.predict_proba().argmax() + 1

    def _trainer(self, mean:float,  mu: float) -> float:
        return mean - np.dot( self.payoffs , self._gibbs_distr(mu) )

    def _gradient(self, mu:float) -> float:
        gibbs = self._gibbs_distr(mu)
        return np.dot( self.payoffs**2 , gibbs ) - np.dot( self.payoffs , gibbs )**2

    def fit(self , mean : float ,
                      max_iter : int = 200 ,
                      tolerance : float = 1e-11 ,
                      verbose : bool = False) -> None:
        mu = 0
        iters = 0
        max_fit = 10
        while abs(self._trainer(mean , mu)) > tolerance:
            mu = mu - self._trainer(mean , mu)/self._gradient(mu)

            if verbose:
                print(f"iteration {iters}:\t{mu}\t{self._trainer(mean , mu)}")
            iters += 1
            if iters > max_iter:
                print(f"Maximum iterations limit {max_iter} exceed.")
                break
        self.multiplier_ = mu



if __name__ == "__main__":
    payoff = lambda i : i
    total_events = 6
    sample_average = 1.2

    model = MaxEnt()
    model.fit(sample_average , verbose = True)
    print(model.predict_proba())
    plt.bar(np.arange(1,total_events+1,1) , model.predict_proba())
    plt.show()
