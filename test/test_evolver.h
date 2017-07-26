#pragma once

#include <memory>
#include "testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_dnn {

TEST(evolver, load_unload_weights) {
    network<sequential> nn;
    core::backend_t backend_type = core::default_engine();
    nn << fully_connected_layer(28 * 28, 80, true, backend_type)
    << sigmoid_layer()
    << fully_connected_layer(80, 10, true, backend_type);

    std::vector<label_t> train_labels; // Remain empty.
    std::vector<vec_t> train_data;     // Remain empty.

    Evolver<sequential> evo(std::make_shared<network<sequential> >(nn),
                            std::make_shared<std::vector<label_t> >(train_labels),
                            std::make_shared<std::vector<vec_t> >(train_data));

    nn.init_weight();

    std::vector<float_t> weight_vector;
    evo.copyInitialGenome(weight_vector);

    int idx = 0;
    for (auto layer : nn) {
        for (auto weights : layer->weights()) {
            for (auto weight : *weights) {
                EXPECT_NEAR(weight_vector[idx], weight, 1e-5);
                idx++;
            }
        }
    }

    // Change all values of the weight_vector to observe a change.
    for (int i = 0; i < weight_vector.size(); i++) {
        weight_vector[i]++;
    }

    Individual indv(std::make_shared<std::vector<float_t>>(weight_vector));
    evo.loadWeights(std::make_shared<Individual>(indv));

    idx = 0;
    for (auto layer : nn) {
        for (auto weights : layer->weights()) {
            for (auto weight : *weights) {
                EXPECT_NEAR(weight_vector[idx], weight, 1e-5);
                idx++;
            }
        }
    }
}

}
