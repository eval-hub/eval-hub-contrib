# Deployed Eval Hub

Running **ibm-clear** on a cluster follows the same **JobSpec** shape as local runs: **`provider_id`** **`ibm-clear`**, **`benchmark_id`**, **`parameters`**, **`model`**, and usually a **data reference** so traces appear inside the job pod. You need a **reachable Eval Hub HTTPS route** (for example an OpenShift **Route** or ingress URL) plus auth; put **`EVALHUB_URL`** and related values in **`examples/.env`** for the notebook or your CI shell.

## What Eval Hub adds

- **Scheduling:** job queue, worker pods, timeouts  
- **Staging traces:** for example **S3** (or compatible object storage) referenced by **`test_data_ref`**; the platform often downloads objects into **`/test_data`** or **`/data`**  
- **Callbacks:** **`callback_url`** for progress (local runs may log connection errors; that is expected)  
- **Credentials:** **`model.auth.secret_ref`** for API keys mounted by the platform  

Exact JSON fields depend on your **Eval Hub version** and operator. Use your deployment’s API docs for **`POST`** job submission.

## Image

Community builds often use **`quay.io/evalhub/community-ibm-clear:latest`** in **`provider.yaml`**. For your own tests you may push a custom image and point the provider definition at **`quay.io/<user>/<repo>:<tag>`** with **`imagePullPolicy: Always`** when iterating.

## Notebook / SDK

For an interactive path (**list providers**, **submit job**, **wait**), open **[`clear_evalhub_example.ipynb`](../clear_evalhub_example.ipynb)** and follow **Part B** after you configure **`examples/.env`** from **`env.example`**.

## Next

[05-benchmarks-and-parameters.md](05-benchmarks-and-parameters.md) — benchmark ids and parameters  
