from utils.plotting import plot_benchmark_results, plot_roc_curves, plot_decision_boundaries_2d

plot_benchmark_results(
    results_df,
    task_type=args.task,
    output_dir=args.plot_dir
)
print(f"[1] Сохранён файл: {args.plot_dir}/benchmark_comparison.png")

if args.task == 'classification':
    plot_roc_curves(
        trained_models_dict=trained_models,
        X_test=X_test,
        y_test=y_test,
        output_dir=args.plot_dir
    )
    print(f"[2] Сохранён файл: {args.plot_dir}/roc_curves.png")

if args.n_features == 2 and args.task == 'classification':
    plot_decision_boundaries_2d(
        trained_models_dict=trained_models,
        X_train=X_train,
        y_train=y_train,
        output_dir=args.plot_dir
    )
    print(f"[3] Сохранён файл: {args.plot_dir}/decision_boundaries.png")
