"""Statistical analysis tools for benchmarking and validation."""

import numpy as np
import scipy.stats as stats
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class SignificanceTest:
    """Statistical significance test result."""
    test_type: str
    statistic: float
    p_value: float
    alpha: float = 0.05
    is_significant: bool = field(init=False)
    effect_size: Optional[float] = None
    
    def __post_init__(self):
        self.is_significant = self.p_value < self.alpha


@dataclass 
class ConfidenceInterval:
    """Confidence interval result."""
    lower: float
    upper: float
    confidence_level: float
    mean: float
    margin_of_error: float = field(init=False)
    
    def __post_init__(self):
        self.margin_of_error = (self.upper - self.lower) / 2


class StatisticalValidator:
    """Validates statistical significance of experimental results."""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        
    def t_test(self, group1: List[float], group2: List[float], 
               paired: bool = False) -> SignificanceTest:
        """Perform t-test between two groups."""
        if paired:
            statistic, p_value = stats.ttest_rel(group1, group2)
            test_type = "paired_t_test"
        else:
            statistic, p_value = stats.ttest_ind(group1, group2)
            test_type = "independent_t_test"
            
        # Calculate Cohen's d effect size
        pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
        effect_size = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
        
        return SignificanceTest(
            test_type=test_type,
            statistic=statistic,
            p_value=p_value,
            alpha=self.alpha,
            effect_size=abs(effect_size)
        )
        
    def wilcoxon_test(self, group1: List[float], group2: List[float]) -> SignificanceTest:
        """Perform Wilcoxon rank-sum test (non-parametric)."""
        statistic, p_value = stats.ranksums(group1, group2)
        
        return SignificanceTest(
            test_type="wilcoxon_ranksum",
            statistic=statistic,
            p_value=p_value,
            alpha=self.alpha
        )
        
    def anova_test(self, *groups: List[float]) -> SignificanceTest:
        """Perform one-way ANOVA test."""
        statistic, p_value = stats.f_oneway(*groups)
        
        return SignificanceTest(
            test_type="one_way_anova",
            statistic=statistic,
            p_value=p_value,
            alpha=self.alpha
        )


class PerformanceComparator:
    """Compares algorithm performance with statistical validation."""
    
    def __init__(self, validator: Optional[StatisticalValidator] = None):
        self.validator = validator or StatisticalValidator()
        
    def compare_algorithms(self, 
                          baseline_results: List[float],
                          test_results: List[float],
                          algorithm_names: Tuple[str, str] = ("Baseline", "Test")) -> Dict[str, Any]:
        """Compare two algorithms with statistical tests."""
        baseline_mean = np.mean(baseline_results)
        test_mean = np.mean(test_results)
        
        # Perform both parametric and non-parametric tests
        t_test = self.validator.t_test(baseline_results, test_results)
        wilcoxon_test = self.validator.wilcoxon_test(baseline_results, test_results)
        
        # Calculate improvement
        improvement = ((test_mean - baseline_mean) / baseline_mean) * 100
        
        return {
            "baseline_mean": baseline_mean,
            "test_mean": test_mean,
            "improvement_percent": improvement,
            "t_test": t_test,
            "wilcoxon_test": wilcoxon_test,
            "baseline_std": np.std(baseline_results),
            "test_std": np.std(test_results),
            "algorithm_names": algorithm_names
        }
        
    def rank_algorithms(self, results: Dict[str, List[float]]) -> List[Tuple[str, float, float]]:
        """Rank algorithms by performance with confidence intervals."""
        rankings = []
        
        for name, scores in results.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            rankings.append((name, mean_score, std_score))
            
        # Sort by mean score (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings


class ConfidenceIntervalCalculator:
    """Calculate confidence intervals for performance metrics."""
    
    def bootstrap_ci(self, data: List[float], 
                    confidence_level: float = 0.95,
                    n_bootstrap: int = 10000) -> ConfidenceInterval:
        """Calculate bootstrap confidence interval."""
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
            
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower = np.percentile(bootstrap_means, lower_percentile)
        upper = np.percentile(bootstrap_means, upper_percentile)
        
        return ConfidenceInterval(
            lower=lower,
            upper=upper,
            confidence_level=confidence_level,
            mean=np.mean(data)
        )
        
    def t_distribution_ci(self, data: List[float], 
                         confidence_level: float = 0.95) -> ConfidenceInterval:
        """Calculate confidence interval using t-distribution."""
        mean = np.mean(data)
        std_err = stats.sem(data)
        alpha = 1 - confidence_level
        
        # Degrees of freedom
        df = len(data) - 1
        
        # Critical t-value
        t_critical = stats.t.ppf(1 - alpha / 2, df)
        
        margin_of_error = t_critical * std_err
        
        return ConfidenceInterval(
            lower=mean - margin_of_error,
            upper=mean + margin_of_error,
            confidence_level=confidence_level,
            mean=mean
        )


def validate_experimental_results(baseline_scores: List[float],
                                treatment_scores: List[float],
                                alpha: float = 0.05) -> Dict[str, Any]:
    """Comprehensive statistical validation of experimental results."""
    validator = StatisticalValidator(alpha=alpha)
    comparator = PerformanceComparator(validator)
    ci_calculator = ConfidenceIntervalCalculator()
    
    # Compare algorithms
    comparison = comparator.compare_algorithms(baseline_scores, treatment_scores)
    
    # Calculate confidence intervals
    baseline_ci = ci_calculator.bootstrap_ci(baseline_scores)
    treatment_ci = ci_calculator.bootstrap_ci(treatment_scores)
    
    # Check for normality (Shapiro-Wilk test)
    baseline_normality = stats.shapiro(baseline_scores)
    treatment_normality = stats.shapiro(treatment_scores)
    
    return {
        "comparison": comparison,
        "baseline_ci": baseline_ci,
        "treatment_ci": treatment_ci,
        "baseline_normality": {
            "statistic": baseline_normality.statistic,
            "p_value": baseline_normality.pvalue,
            "is_normal": baseline_normality.pvalue > alpha
        },
        "treatment_normality": {
            "statistic": treatment_normality.statistic,
            "p_value": treatment_normality.pvalue,
            "is_normal": treatment_normality.pvalue > alpha
        },
        "sample_sizes": {
            "baseline": len(baseline_scores),
            "treatment": len(treatment_scores)
        }
    }