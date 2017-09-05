// Takes some inspiration from examples/mnist/train.cpp

#include <iostream>
#include <memory>
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;

static void construct_simple_net(network<sequential> &nn,
                          core::backend_t backend_type) {
    // Baby network for now...
    nn << fully_connected_layer(28 * 28, 80, true, backend_type)
       << sigmoid_layer()
       << fully_connected_layer(80, 10, true, backend_type);
}

static void leea_experiment(const std::string &data_path, const int seed) {
    Random * random = new Random(seed);

    network<sequential> nn;
    core::backend_t backend_type = core::default_engine();
    size_t num_classes = 10;

    construct_simple_net(nn, backend_type);

    std::cout << "Loading mnist data..." << std::endl;

    std::vector<vec_t> one_hot_labels;
    std::vector<label_t> test_labels; // Test labels can be in non-one-hot style.
    std::vector<vec_t> train_images, test_images;

    parse_mnist_labels(data_path + "/train-labels.idx1-ubyte", &one_hot_labels, num_classes);
    parse_mnist_images(data_path + "/train-images.idx3-ubyte", &train_images, -1.0, 1.0, 0, 0); // Skip the padding.

    parse_mnist_labels(data_path + "/t10k-labels.idx1-ubyte", &test_labels);
    parse_mnist_images(data_path + "/t10k-images.idx3-ubyte", &test_images, -1.0, 1.0, 0, 0); // Skip the padding.

    std::cout << "Start training..." << std::endl;

    Evolver<sequential, mse> evo(std::make_shared<network<sequential> >(nn),
                            std::make_shared<std::vector<vec_t> >(one_hot_labels),
                            std::make_shared<std::vector<vec_t> >(train_images),
                            random);

    evo.evolve();
    delete random;
}

static void usage(const char *argv0) {
    std::cout << "Usage: " << argv0 << " --data_path path_to_dataset_folder \n"
            << "\t--seed 0" <<
            std::endl;
}

int main(int argc, char **argv) {
    std::string data_path = "";
    int seed = 0;

    if (argc == 2) {
        std::string argname(argv[1]);
        if (argname == "--help" || argname == "-h") {
            usage(argv[0]);
            return 0;
        }
    }

    for (int count = 1; count + 1 < argc; count += 2) {
        std::string argname(argv[count]);
        if (argname == "--data_path") {
            data_path = std::string(argv[count + 1]);
        }
        else if (argname == "--seed") {
            seed = atoi(argv[count + 1]);
        }
        else {
          std::cerr << "Invalid parameter specified - \"" << argname << "\""
                    << std::endl;
          usage(argv[0]);
          return -1;
        }
    }

    if (data_path == "") {
        std::cerr << "Data path not specified." << std::endl;
        usage(argv[0]);
        return -1;
    }

    std::cout << "Running with the following parameters: " << std::endl
            << "Data path: " << data_path << std::endl
            << "Seed: " << seed << std::endl
            << std::endl;

    try {
        leea_experiment(data_path, seed);
    }
    catch (tiny_dnn::nn_error &err) {
        std::cerr << "Exception: " << err.what() << std::endl;
    }

    return 0;
}
