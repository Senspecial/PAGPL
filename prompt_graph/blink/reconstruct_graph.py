from prompt_graph.blink.blink import Server


def reconstruct_graph(data, epsilon, delta, gnn_type, pre_train_model_path, hid_dim, num_layer, epochs, lr, reg_weight, gnn_weight):
    server = Server(data, epsilon, delta)
    prior = server.estimate_prior(gnn_type, pre_train_model_path, hid_dim, num_layer, epochs, lr, reg_weight, gnn_weight)
    posterior = server.estimate_posterior(prior)
    reconstruct_graph = server.reconstruct_graph(posterior)

    return reconstruct_graph
