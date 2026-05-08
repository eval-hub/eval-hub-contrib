# Overview: CLEAR, Eval Hub, and this adapter

## IBM CLEAR

**CLEAR** ([IBM/CLEAR](https://github.com/IBM/CLEAR)) stands for **Comprehensive LLM Error Analysis and Reporting**. It accepts **agent traces** as input, usually JSON from instrumentation that matches MLflow conventions. CLEAR runs an **LLM as judge** pipeline, clusters recurring problems, and writes structured results. The main outputs are **`clear_results.json`** and a **static HTML** dashboard such as **`clear_results.html`**.

**Typical use with Eval Hub:** you submit evaluation jobs; the **`ibm-clear`** adapter runs CLEAR on traces (often staged from object storage) and returns metrics to the platform. **Local use:** you run the same CLEAR pipeline with **`python main.py`** and a JobSpec JSON at **`EVALHUB_JOB_SPEC_PATH`**, so you can develop and debug without a cluster while using the same fields Eval Hub sends on the wire.

## Eval Hub

**Eval Hub** is a **platform** that **schedules evaluation jobs** (for example on Kubernetes): you submit a **job specification** (model, benchmark, optional data references), and workers run **provider adapters** and return **metrics** and status.

The eval-hub-contrib repository holds **community adapters**. **IBM CLEAR** is **one** of those providers; its adapter id is **`ibm-clear`**.

## The IBM CLEAR adapter (`adapters/clear`)

This directory implements **`ClearAdapter`**: it turns Eval Hub’s **JobSpec** into CLEAR’s **agentic** pipeline, then reads CLEAR’s **outputs**, **`clear_results.json`** (metrics source) and the **HTML** report CLEAR wrote. It maps statistics into Eval Hub **metrics**, preserves or restyles HTML for artifacts, reports progress, and optionally uploads to **MLflow** or **OCI**.

Eval Hub **orchestrates** jobs; **ibm-clear** is **one** adapter alongside others; **CLEAR** is the **analysis engine** running inside this adapter.

## What to read next

New to this repo: follow **[`examples/README.md`](../README.md)** (ordered docs, sample inputs/outputs, optional notebook). Then use [02-agent-traces.md](02-agent-traces.md), [03-local-run.md](03-local-run.md), and [05-benchmarks-and-parameters.md](05-benchmarks-and-parameters.md) as needed.
