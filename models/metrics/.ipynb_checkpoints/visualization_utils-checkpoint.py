                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                , max_num_clients, figsize=(15, 12), title_fontsize=16, max_name_len=10, **kwargs):
    """Plots the clients' test accuracy vs. the round number.

    Args:
        stat_metrics: pd.DataFrame as written by writer.py.
        sys_metrics: pd.DataFrame as written by writer.py. Allows us to know which client actually performed training
            in each round. If None, then no indication is given of when was each client trained.
        max_num_clients: Maximum number of clients to plot.
        figsize: Size of the plot as specified by plt.figure().
        title_fontsize: Font size for the plot's title.
        max_name_len: Maximum length for a client's id.
        kwargs: Arguments to be passed to _set_plot_properties."""
    # Plot accuracies per client.
    clients = stat_metrics[CLIENT_ID_KEY].unique()[:max_num_clients]
    cmap = plt.get_cmap('jet_r')
    plt.figure(figsize=figsize)

    for i, c in enumerate(clients):
        color = cmap(float(i) / len(clients))
        c_accuracies = stat_metrics.loc[stat_metrics[CLIENT_ID_KEY] == c]
        plt.plot(c_accuracies[NUM_ROUND_KEY], c_accuracies[ACCURACY_KEY], color=color)

    plt.suptitle('Accuracy vs Round Number (Per Client)', fontsize=title_fontsize)
    plt.title('(Dots indicate that client was trained at that round)')
    plt.xlabel('Round Number')
    plt.ylabel('Accuracy')

    labels = stat_metrics[[CLIENT_ID_KEY, NUM_SAMPLES_KEY]].drop_duplicates()
    labels = labels.loc[labels[CLIENT_ID_KEY].isin(clients)]
    labels = ['%s, %d' % (row[CLIENT_ID_KEY][:max_name_len], row[NUM_SAMPLES_KEY])
              for _, row in labels.iterrows()]
    plt.legend(labels, title='client id, num_samples', loc='upper left')

    # Plot moments in which the clients were actually used for training.
    # To do this, we need the system metrics (to know which client actually performed training in each round).
    if sys_metrics is not None:
        for i, c in enumerate(clients[:max_num_clients]):
            color = cmap(float(i) / len(clients))
            c_accuracies = stat_metrics.loc[stat_metrics[CLIENT_ID_KEY] == c]
            c_computation = sys_metrics.loc[sys_metrics[CLIENT_ID_KEY] == c]
            c_join = pd.merge(c_accuracies, c_computation, on=NUM_ROUND_KEY, how='inner')
            if not c_join.empty:
                plt.plot(
                    c_join[NUM_ROUND_KEY],
                    c_join[ACCURACY_KEY],
                    linestyle='None',
                    marker='.',
                    color=color,
                    markersize=18)

    _set_plot_properties(kwargs)
    plt.show()


def plot_bytes_written_and_read(sys_metrics, rolling_window=10, figsize=(10, 8), title_fontsize=16, **kwargs):
    """Plots the cumulative sum of the bytes written and read by the server.

    Args:
        sys_metrics: pd.DataFrame as written by writer.py.
        rolling_window: Number of previous rounds to consider in the cumulative sum.
        figsize: Size of the plot as specified by plt.figure().
        title_fontsize: Font size for the plot's title.
        kwargs: Arguments to be passed to _set_plot_properties."""

    plt.figure(figsize=figsize)

    server_metrics = sys_metrics.groupby(NUM_ROUND_KEY, as_index=False).sum()
    rounds = server_metrics[NUM_ROUND_KEY]
    server_metrics = server_metrics.rolling(rolling_window, on=NUM_ROUND_KEY, min_periods=1).sum()
    plt.plot(rounds, server_metrics['bytes_written'], alpha=0.7)
    plt.plot(rounds, server_metrics['bytes_read'], alpha=0.7)

    plt.title('Bytes Written and Read by Server vs. Round Number', fontsize=title_fontsize)
    plt.xlabel('Round Number')
    plt.ylabel('Bytes')
    plt.legend(['Bytes Written', 'Bytes Read'], loc='upper left')
    _set_plot_properties(kwargs)
    plt.show()


def plot_client_computations_vs_round_number(
        sys_metrics,
        aggregate_window=20,
        max_num_clients=20,
        figsize=(25, 15),
        title_fontsize=16,
        max_name_len=10,
        range_rounds=None):
    """Plots the clients' local computations against round number.

    Args:
        sys_metrics: pd.DataFrame as written by writer.py.
        aggregate_window: Number of rounds that are aggregated. e.g. If set to 20, then
            rounds 0-19, 20-39, etc. will be added together.
        max_num_clients: Maximum number of clients to plot.
        figsize: Size of the plot as specified by plt.figure().
        title_fontsize: Font size for the plot's title.
        max_name_len: Maximum length for a client's id.
        range_rounds: Tuple representing the range of rounds to be plotted. The rounds
            are subsampled before aggregation. If None, all rounds are considered."""
    plt.figure(figsize=figsize)

    num_rounds = sys_metrics[NUM_ROUND_KEY].max()
    clients = sys_metrics[CLIENT_ID_KEY].unique()[:max_num_clients]

    comp_matrix = []
    matrix_keys = [c[:max_name_len] for c in clients]

    for c in clients:
        client_rows = sys_metrics.loc[sys_metrics[CLIENT_ID_KEY] == c]
        client_rows = client_rows.groupby(NUM_ROUND_KEY, as_index=False).sum()
        client_computations = [0 for _ in range(num_rounds)]

        for i in range(num_rounds):
            computation_row = client_rows.loc[client_rows[NUM_ROUND_KEY] == i]
            if not computation_row.empty:
                client_computations[i] = computation_row.iloc[0][LOCAL_COMPUTATIONS_KEY]
        comp_matrix.append(client_computations)

    if range_rounds:
        assert range_rounds[0] >= 0 and range_rounds[1] > 0
        assert range_rounds[0] <= range_rounds[1]
        assert range_rounds[0] < len(comp_matrix[0]) and range_rounds[1] < len(comp_matrix[0]) + 1
        assert range_rounds[1] - range_rounds[0] >= aggregate_window
        new_comp_matrix = []
        for i in range(len(comp_matrix)):
            new_comp_matrix.append(comp_matrix[i][range_rounds[0]:range_rounds[1]])
        comp_matrix = new_comp_matrix

    agg_comp_matrix = []
    for c_comp_vals in comp_matrix:
        num_rounds = len(c_comp_vals)
        agg_c_comp_vals = []
        for i in range(num_rounds // aggregate_window):
            agg_c_comp_vals.append(
                np.sum(c_comp_vals[i * aggregate_window:(i + 1) * aggregate_window]))
        agg_comp_matrix.append(agg_c_comp_vals)

    plt.title(
        'Total Client Computations (FLOPs) vs. Round Number (x %d)' % aggregate_window,
        fontsize=title_fontsize)
    im = plt.imshow(agg_comp_matrix)
    plt.yticks(range(len(matrix_keys)), matrix_keys)
    plt.colorbar(im, fraction=0.02, pad=0.01)
    plt.show()


def get_longest_flops_path(sys_metrics):
    """Prints the largest amount of flops required to complete training.

    To calculate this metric, we:
        1. For each round, pick the client that required the largest amount
            of local training.
        2. Sum the FLOPS from the clients picked in step 1 across rounds.

    TODO: This metric would make more sense with seconds instead of FLOPS.

    Args:
        sys_metrics: pd.DataFrame as written by writer.py."""
    num_rounds = sys_metrics[NUM_ROUND_KEY].max()
    clients = sys_metrics[CLIENT_ID_KEY].unique()

    comp_matrix = []

    for c in clients:
        client_rows = sys_metrics.loc[sys_metrics[CLIENT_ID_KEY] == c]
        client_rows = client_rows.groupby(NUM_ROUND_KEY, as_index=False).sum()
        client_computations = [0 for _ in range(num_rounds)]

        for i in range(num_rounds):
            computation_row = client_rows.loc[client_rows[NUM_ROUND_KEY] == i]
            if not computation_row.empty:
                client_computations[i] = computation_row.iloc[0][LOCAL_COMPUTATIONS_KEY]
        comp_matrix.append(client_computations)

    comp_matrix = np.asarray(comp_matrix)
    num_flops = np.sum(np.max(comp_matrix, axis=0))
    return '%.2E' % Decimal(num_flops.item())