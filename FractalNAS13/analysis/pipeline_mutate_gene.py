# The handcrafted mutate_gene the audited pipeline ran, verbatim from
# ab/gpt/brute/ga/meta_evolution/genetic_algorithm_baseline.py (nn-gpt, as of the October 2026
# runs). _mutate() offers each gene to it with probability mutation_rate = 0.2, and it redraws
# the gene with probability 2 x 0.2 = 0.4: an effective per-gene rate of 0.08. Every LLM-written
# mutate_gene replaced this function, and is replayed against it.
def mutate_gene(self, gene_name, current_value, possible_values):
    if possible_values:
        choices = list(set(possible_values) - set([current_value]))
        if choices:
            mutation_probability = self.mutation_rate * 2  # Increase mutation probability
            if random.random() < mutation_probability:
                return random.choice(choices)
    return current_value
