---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Citation Key: "{{citekey}}"
Paper Link: "{{url}}"
Year: "{{date | format(\"YYYY\")}}"
Reading Status: To-Read
Tier: Recommended
Topic: Research-Paper
Authors: "{% for creator in creators %}{{creator.firstName}} {{creator.lastName}}{% if not loop.last %}, {% endif %}{% endfor %}"
tags:
  - Research-Paper
---

# {{title}}

> [!info] Reference
> {{bibliography}}
>
> [Open in Zotero]({{desktopURI}}) · Citation key: `{{citekey}}`

{% if abstractNote %}
## Abstract

{{abstractNote}}
{% endif %}

## Reading objective

{% persist "reading-objective" %}
- [ ] What is the paper's precise task and novelty?
- [ ] Which data, supervision, metrics, and splits does it use?
- [ ] Which component, baseline, or failure mode affects [[CVPR 2027 Submission Roadmap — Temporal Self-Contact]]?
{% endpersist %}

## My synthesis

{% persist "synthesis" %}

{% endpersist %}

## Imported annotations

{% persist "annotations" %}
{% set newAnnotations = annotations | filterby("date", "dateafter", lastImportDate) %}
{% for annotation in newAnnotations %}
{% if annotation.annotatedText %}
> [!quote] Page {{annotation.pageLabel}}
> {{annotation.annotatedText}}
{% endif %}
{% if annotation.comment %}
**Comment:** {{annotation.comment}}
{% endif %}

{% endfor %}
{% endpersist %}

## Research connection

{% persist "research-connection" %}
- Related claim:
- Baseline/component to reproduce:
- Citation location in draft:
{% endpersist %}

