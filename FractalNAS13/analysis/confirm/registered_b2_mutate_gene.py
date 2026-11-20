# cifar10 deepseek run2 meta-iteration 2 | inner-GA best 3-epoch acc 84.05 | reward 8.72
    def mutate_gene(self, gene_name, current_value, possible_values):
        if possible_values:
            choices = list(set(possible_values) - set([current_value]))
            if choices:
                mutation_probability = self.mutation_rate * 2  # Increase mutation probability
                if random.random() < mutation_probability:
                    # Aggressively choose a new value that is far from the current one
                    new_value = random.choice(choices)
                    while new_value == current_value:
                        new_value = random.choice(choices)
                    return new_value
        return current_value
