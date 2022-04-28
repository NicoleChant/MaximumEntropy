from dataclasses import dataclass , field
from typing import Callable , List
import numpy as np


@dataclass
class MaxEnt:

    total_events:int = field(default = 3)
    payoff : Callable[ [int] , float] = field(default = lambda i : i)
    multiplier_: float = field(init = False , default = None)
    payoffs : np.ndarray = field(init = False)

    def __post_init__(self) -> None:
        self.payoffs = self.payoff(np.arange(1,self.total_events+1,1))

    @property
    def predict_proba(self) -> np.ndarray:
        try:
            return self._gibbs_distr(self.multiplier_)
        except Exception as e:
            raise Exception("Model is not fitted!")

    def _gibbs_distr(self , mu) -> np.ndarray:
        partition_func_eval = self._partition_function(mu)
        return np.e**( self.payoffs * mu)/partition_func_eval

    def _partition_function(self,mu) -> float:
        return (np.e**(self.payoffs * mu)).sum()

    def predict(self) -> int:
        return self.probabilities.argmax()

    def _trainer(self, mean:float,  mu: float) -> float:
        return mean - np.dot( self.payoffs , self._gibbs_distr(mu) )

    def _gradient(self, mu:float) -> float:
        gibbs = self._gibbs_distr(mu)
        return -  np.dot( self.payoffs**2 , gibbs ) + np.dot( self.payoffs , gibbs )**2

    def fit(self , mean : float ,
                      max_iter : int = 200 ,
                      tolerance : float = 1e-10 ,
                      verbose : bool = False) -> bool:
        mu = 0
        iters = 0
        while abs(self._trainer(mean , mu)) > tolerance:
            mu = mu - self._trainer(mean , mu)/self._gradient(mu)
            if verbose:
                print(f"iteration {iters}:\t{mu}\t{self._trainer(mean , mu)}")
            iters += 1
            if iters > max_iter:
                print(f"Maximum iterations limit {max_iter} exceed.")
                break
        self.multiplier_ = mu


class RandomProcess:

    def generate(self):
        roll = random.random()
        if roll <= 0.5:
            return 100
        elif roll <= 0.8:
            return 220
        else:
            return 450


if __name__ == "__main__":
    payoff = lambda i : i
    sample_average = 3.5
    total_events = 6
    model = MaxEnt(sample_average , total_events , payoff)
    model.fit(verbose = True)
    print(model.probabilities)
    plt.bar(np.arange(1,total_events+1,1) , model.probabilities)
    plt.show()
