/**
 * This should 100% be comthing that reads the params.config file.
 * I can't be asked to make/find a file reader though :)
 */
#pragma once

namespace tiny_dnn {
    struct Params {
      Params() = delete;

      static const size_t population_size = 1000;
      static const size_t max_generations = 20;
      static const size_t sample_count = 2; //< 2 examples each generation.
      static constexpr float mutation_power = 0.03; //< Maximum range of mutation.
      /// Power decayed gradually, 0 disables,
      /// 1 will leave 0 mutation power at last generation.
      static constexpr float mutation_power_decay = 0.99;
      static constexpr float mutation_rate = 0.04; //< Proportion of weights to mutate.
      static constexpr float mutation_rate_decay = 0; //< Decay rate of mutation, per generation.
      /// Proportion of offspring produced by sexual reproduction.
      static constexpr float sex_proportion = 0.5;
      /// Top X proportion of individuals selected for reproduction.
      static constexpr float selection_proportion = 0.4;
      static constexpr float initial_weights_delta = 1.0; //< Initial weights between [-Wd, Wd]
      static constexpr float fitness_decay_rate = 0.2; //< 0.2 = 20% decay per evaluation.
      static const size_t tracking_stride = 1000; // Every N generations, print out info.
      static constexpr float min_fitness = 0.00001;
    };
}
