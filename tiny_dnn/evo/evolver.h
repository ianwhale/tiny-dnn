#pragma once

#include <memory>
#include <vector>
#include <algorithm>
#include <iostream>
#include "tiny_dnn/evo/params.h"
#include "tiny_dnn/evo/individual.h"
#include "tiny_dnn/util/util.h"

namespace tiny_dnn {
    template <typename NetType>
    /**
     * Evolver class. Aims to minimize the weights of the network using LEEA.
     */
    class Evolver {
    public:
        Evolver(std::shared_ptr<network<NetType>> nn,
                std::shared_ptr<std::vector<label_t>> train_labels,
                std::shared_ptr<std::vector<vec_t>> train_data) : mNetwork(nn) {

            // Initialize population.
            // TODO: Parallelize.
            printNetwork();
        }


        void printNetwork() {
            int bungus = 0;
            for (auto layer : *mNetwork) {
                for (auto weights : layer->weights()) {
                    for (auto weight : *weights) {
                        std::cout << weight << std::endl;
                    }
                }
            }
        }

        /**
         * Main evolution loop.
         * Evaluate to find best individuals and reproduce best.
         *
         */
        template <typename Error> // Error function to use.
        void evolve() {
            while (mCurrentGeneration < Params::max_generations) {
                evaluate<Error>();
                sortPopulation();
                reproduce();
                mCurrentGeneration++;
            }
        }

        /**
         * Assign a fitness to each Individual by evaluating the network and
         * getting the value of the loss function for a minibatch.
         */
        template <typename Error>
        void evaluate() {
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
        void reproduce() {
            // TODO: Parallelize.
        }

    protected:
        int mCurrentGeneration = 0;
        std::vector<std::shared_ptr<Individual>> mPopulation;
        std::shared_ptr<network<NetType>> mNetwork;
        std::shared_ptr<std::vector<label_t>> mTrainLabels;
        std::shared_ptr<std::vector<vec_t>> mTrainData;
    };

}
