"""DeepEval adapter for eval-hub.

Module layout, in dependency order:

``supplementary``    extra aggregate metrics derived from the per-case scores
``metrics``          DeepEval metric factories (one callable per benchmark family)
``benchmarks``       ``BenchmarkSpec`` and the ``BENCHMARKS`` registry: the one
                     place that knows what each benchmark needs and produces
``data_loading``     dataset discovery and CSV / JSONL / JSON reading
``cases``            records -> DeepEval test cases
``judge``            judge model construction
``json_correctness`` direct JSON / JSON Schema validation (no judge, no metric)
``results``          DeepEval output -> eval-hub results and EvalCard entries
``settings``         job parameter parsing and validation
``reporting``        EvalCard, results summary, OCI artifact and MLflow
``adapter``          ``DeepEvalAdapter``, which sequences the job phases
"""
