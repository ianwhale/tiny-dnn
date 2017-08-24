#pragma once

#include <memory>
#include <iostream>
#include <algorithm>
#include "testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_dnn {

// Puts 10 random data points into the data pointer.
void put_random_data(std::vector<vec_t> *data, size_t columns) {
    data->resize(10);
    for (size_t i = 0; i < 10; i++) {
        for (size_t j = 0; j < columns; j++) {
            (*data)[i].push_back(uniform_rand(float_t(0), float_t(1)));
        }
    }
}

TEST(evolver, initialize_population) {
    network<sequential> nn;
    core::backend_t backend_type = core::default_engine();
    nn << fully_connected_layer(5, 2, true, backend_type)
    << sigmoid_layer()
    << fully_connected_layer(2, 1, true, backend_type);

    std::vector<vec_t> train_labels;
    std::vector<vec_t> train_data;

    put_random_data(&train_labels, 1);
    put_random_data(&train_data, 5);

    Evolver<sequential, mse> evo(std::make_shared<network<sequential>>(nn),
                            std::make_shared<std::vector<vec_t>>(train_labels),
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
    nn << fully_connected_layer(5, 2, true, backend_type)
    << sigmoid_layer()
    << fully_connected_layer(2, 1, true, backend_type);

    std::vector<vec_t> train_labels;
    std::vector<vec_t> train_data;

    put_random_data(&train_labels, 1);
    put_random_data(&train_data, 5);

    Evolver<sequential, mse> evo(std::make_shared<network<sequential>>(nn),
                            std::make_shared<std::vector<vec_t>>(train_labels),
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
    nn << fully_connected_layer(5, 2, true, backend_type)
    << sigmoid_layer()
    << fully_connected_layer(2, 1, true, backend_type);

    std::vector<vec_t> train_labels;
    std::vector<vec_t> train_data;

    put_random_data(&train_labels, 1);
    put_random_data(&train_data, 5);

    Evolver<sequential, mse> evo(std::make_shared<network<sequential>>(nn),
                            std::make_shared<std::vector<vec_t>>(train_labels),
                            std::make_shared<std::vector<vec_t>>(train_data));

    auto pre_repro = evo.getPopulation();

    EXPECT_NO_THROW(evo.reproducePopulation());

    auto post_repro = evo.getPopulation();

    EXPECT_EQ(pre_repro->size(), post_repro->size());

    // Don't know how to test anything else...
}

TEST(evolver, mini_batch_handler) {
    network<sequential> nn;
    core::backend_t backend_type = core::default_engine();
    nn << fully_connected_layer(5, 2, true, backend_type)
    << sigmoid_layer()
    << fully_connected_layer(2, 1, true, backend_type);

    std::vector<vec_t> train_labels;
    std::vector<vec_t> train_data;

    vec_t dummy(5, float_t(0));
    vec_t dummy_label(1, float_t(0));
    // One more than sample count to ensure we get a wrap around.
    for (size_t i = 0; i < Params::sample_count + 1; i++) {
        train_labels.push_back(dummy_label);
        train_data.push_back(dummy);

        // Something to differentiate the samples.
        train_labels[i][0] = i;
        train_data[i][i] = float_t(1);
    }

    Evolver<sequential, mse> evo(std::make_shared<network<sequential>>(nn),
                            std::make_shared<std::vector<vec_t>>(train_labels),
                            std::make_shared<std::vector<vec_t>>(train_data));

    // The population is evaluated on evolver construction so we should expect
    // a rollover into the next epoch on the this batch.

    std::vector<vec_t> mini_labels;
    std::vector<vec_t> mini_data;

    evo.getMiniBatchHandler()->nextBatch(&mini_labels, &mini_data);

    EXPECT_EQ(train_labels[train_labels.size() - 1], mini_labels[0]);
    EXPECT_EQ(train_data[train_data.size() - 1], mini_data[0]);

    for (size_t i = 1; i < mini_labels.size(); i++) {
        EXPECT_EQ(train_labels[i - 1], mini_labels[i]);
        EXPECT_EQ(train_data[i - 1], mini_data[i]);
    }
}

}
