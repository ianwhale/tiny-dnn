#pragma once

#include <memory>
#include <vector>
#include <algorithm>
#include <iostream>
#include <typeinfo>
#include "tiny_dnn/evo/params.h"
#include "tiny_dnn/evo/individual.h"
#include "tiny_dnn/util/util.h"

namespace tiny_dnn {

typedef std::shared_ptr<std::vector<std::shared_ptr<Individual>>> pop_ptr;

    /**
     * Evolver class. Aims to minimize the weights of the network using LEEA.
     */
    template <typename NetType>
    class Evolver {
    public:
        Evolver(std::shared_ptr<network<NetType>> nn,
                std::shared_ptr<std::vector<label_t>> train_labels,
                std::shared_ptr<std::vector<vec_t>> train_data)
                : mNetwork(nn) {

            for (size_t i = 0; i < Params::population_size; i++) {
                mPopulation.push_back(nullptr);
            }

            initializePopulation(getWeightCount());
            // TODO: Parallelize.
            // std::vector<float> initial_genome;
            // copyInitialGenome(initial_genome);

            // Only need the length of the new genome.
            // Copy initial is only there for historical reasons now I guess...

            // Emplace back shared pointers for initial genome.
        }

        /**
         * Initialize the population in parallel.
         * @param genome_length
         */
        void initializePopulation(size_t genome_length) {
            for_i(true, Params::population_size, [&](size_t i) {
                mPopulation[i] = std::make_shared<Individual>(genome_length);
            });
        }

        /**
         * Get how many weights are in the network.
         * @return count
         */
        size_t getWeightCount() {
            size_t count = 0;
            for (auto layer : *mNetwork) {
                for (auto weights : layer->weights()) {
                    for (auto weight : *weights) {
                        count++;
                    }
                }
            }
            return count;
        }

        /**
         * Used for testing...
         * @param initial_genome
         */
        void copyInitialGenome(std::vector<float> &initial_genome) {
            initial_genome.empty(); // Just to make sure...
            for (auto layer : *mNetwork) {
                for (auto weights : layer->weights()) {
                    for (auto weight : *weights) {
                        initial_genome.push_back(weight);
                    }
                }
            }
        }

        /**
         * Loads a network's weights with an individual's genome.
         */
        void loadWeights(std::shared_ptr<Individual> individual) {
            int idx = 0;
            for (auto layer : *mNetwork) {
                layer->load(*(individual->getGenome()), idx);
            }
        }

        /**
         * Main evolution loop.
         * Evaluate to find best individuals and reproduce best.
         */
        template <typename Error> // Error function to use.
        void evolve() {
            while (mCurrentGeneration < Params::max_generations) {
                evaluatePopulation<Error>();
                sortPopulation();
                reproducePopulation();
                mCurrentGeneration++;
            }
        }

        /**
         * Assign a fitness to each Individual by evaluating the network and
         * getting the value of the loss function for a minibatch.
         */
        template <typename Error>
        void evaluatePopulation() {
            // TODO: Parallelize.
        }

        /**
         * Sort the population by fitness.
         */
        inline void sortPopulation() {
            std::sort(mPopulation.begin(), mPopulation.end(),
                    [](const std::shared_ptr<Individual> a,
                       const std::shared_ptr<Individual> b) -> bool
                   {   // Sort in descending order.
                       return a->getFitness() > b->getFitness();
                   });
        }

        /**
         * Reproduce the best individuals based on Params::selection_proportion.
         */
        void reproducePopulation() {
            // TODO: Parallelize.
        }

        pop_ptr getPopulation() {
            return std::make_shared<std::vector<std::shared_ptr<Individual>>>
                    (mPopulation);
        }

    protected:
        int mCurrentGeneration = 0;
        std::vector<std::shared_ptr<Individual>> mPopulation;
        std::shared_ptr<network<NetType>> mNetwork;
        std::shared_ptr<std::vector<label_t>> mTrainLabels;
        std::shared_ptr<std::vector<vec_t>> mTrainData;
    };

}
