#pragma once

#include <memory>
#include <iostream>
#include <algorithm>
#include "testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_dnn {

TEST(evolver, initialize_population) {
    network<sequential> nn;
    core::backend_t backend_type = core::default_engine();
    nn << fully_connected_layer(28 * 28, 80, true, backend_type)
    << sigmoid_layer()
    << fully_connected_layer(80, 10, true, backend_type);

    std::vector<label_t> train_labels; // Remain empty.
    std::vector<vec_t> train_data;     // Remain empty.

    Evolver<sequential> evo(std::make_shared<network<sequential>>(nn),
                            std::make_shared<std::vector<label_t>>(train_labels),
                            std::make_shared<std::vector<vec_t>>(train_data));

    size_t size = evo.getWeightCount();
    pop_ptr population = evo.getPopulation();

    for (auto individual : *population) {
        auto genome = *(individual->getGenome());
        EXPECT_EQ(size, genome.size());

        for (auto weight : genome) {
            EXPECT_TRUE(weight <= Params::initial_weights_delta
                        && weight >= (-1 * Params::initial_weights_delta));
        }
    }
}

TEST(evolver, load_unload_weights) {
    network<sequential> nn;
    core::backend_t backend_type = core::default_engine();
    nn << fully_connected_layer(28 * 28, 80, true, backend_type)
    << sigmoid_layer()
    << fully_connected_layer(80, 10, true, backend_type);

    std::vector<label_t> train_labels; // Remain empty.
    std::vector<vec_t> train_data;     // Remain empty.

    Evolver<sequential> evo(std::make_shared<network<sequential>>(nn),
                            std::make_shared<std::vector<label_t>>(train_labels),
                            std::make_shared<std::vector<vec_t>>(train_data));

    size_t weight_count = evo.getWeightCount();

    std::shared_ptr<Individual> indv_ptr(new Individual(weight_count));

    evo.loadWeights(indv_ptr);
    auto network_weights = *(evo.getCurrentNetworkWeights());
    auto genome = *(indv_ptr->getGenome());

    for (size_t i = 0; i < weight_count; i++) {
        EXPECT_EQ(genome[i], network_weights[i]);
    }

    genome = *(indv_ptr->getGenome());

    for (auto & weight : genome) {
        weight++;
    }

    indv_ptr->setGenome(genome);

    evo.loadWeights(indv_ptr);
    network_weights = *(evo.getCurrentNetworkWeights());

    for (size_t i = 0; i < weight_count; i++) {
        EXPECT_EQ(genome[i], network_weights[i]);
    }
}

TEST(evolver, reproduce_population) {
    network<sequential> nn;
    core::backend_t backend_type = core::default_engine();
    nn << fully_connected_layer(28 * 28, 80, true, backend_type)
    << sigmoid_layer()
    << fully_connected_layer(80, 10, true, backend_type);

    std::vector<label_t> train_labels; // Remain empty.
    std::vector<vec_t> train_data;     // Remain empty.

    Evolver<sequential> evo(std::make_shared<network<sequential>>(nn),
                            std::make_shared<std::vector<label_t>>(train_labels),
                            std::make_shared<std::vector<vec_t>>(train_data));

    EXPECT_NO_THROW(evo.reproducePopulation());

}

}
