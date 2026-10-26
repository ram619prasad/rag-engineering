
from pathlib import Path
import json

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "engineering_kb"
    / "documents"
)


DOCUMENTS = {
    # ==========================================
    # KUBERNETES — 6 documents
    # ==========================================
    "kubernetes_resource_management.md": (
        "https://raw.githubusercontent.com/kubernetes/website/main/"
        "content/en/docs/concepts/configuration/manage-resources-containers.md"
    ),
    "kubernetes_hpa.md": (
        "https://raw.githubusercontent.com/kubernetes/website/main/"
        "content/en/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale.md"
    ),
    "kubernetes_pod_scheduling.md": (
        "https://raw.githubusercontent.com/kubernetes/website/main/"
        "content/en/docs/concepts/scheduling-eviction/assign-pod-node.md"
    ),
    "kubernetes_services.md": (
        "https://raw.githubusercontent.com/kubernetes/website/main/"
        "content/en/docs/concepts/services-networking/service.md"
    ),
    "kubernetes_persistent_volumes.md": (
        "https://raw.githubusercontent.com/kubernetes/website/main/"
        "content/en/docs/concepts/storage/persistent-volumes.md"
    ),
    "kubernetes_resource_quotas.md": (
        "https://raw.githubusercontent.com/kubernetes/website/main/"
        "content/en/docs/concepts/policy/resource-quotas.md"
    ),

    # ==========================================
    # DOCKER — 2 documents
    # ==========================================
    "docker_dockerfile.md": (
        "https://raw.githubusercontent.com/docker/docs/main/"
        "content/get-started/docker-concepts/building-images/"
        "writing-a-dockerfile.md"
    ),
    "docker_multistage.md": (
        "https://raw.githubusercontent.com/docker/docs/main/"
        "content/get-started/docker-concepts/building-images/"
        "multi-stage-builds.md"
    ),

    # ==========================================
    # PROMETHEUS — 4 documents
    # ==========================================
    "prometheus_data_model.md": (
        "https://raw.githubusercontent.com/prometheus/docs/main/"
        "docs/concepts/data_model.md"
    ),
    "prometheus_metric_types.md": (
        "https://raw.githubusercontent.com/prometheus/docs/main/"
        "docs/concepts/metric_types.md"
    ),
    "prometheus_instrumentation.md": (
        "https://raw.githubusercontent.com/prometheus/docs/main/"
        "docs/practices/instrumentation.md"
    ),
    "prometheus_histograms.md": (
        "https://raw.githubusercontent.com/prometheus/docs/main/"
        "docs/practices/histograms.md"
    ),

    # ==========================================
    # DISTRIBUTED SYSTEMS — 5 documents
    # ==========================================
    "distributed_retry.md": (
        "https://raw.githubusercontent.com/MicrosoftDocs/"
        "architecture-center/main/docs/patterns/retry-content.md"
    ),
    "distributed_circuit_breaker.md": (
        "https://raw.githubusercontent.com/MicrosoftDocs/"
        "architecture-center/main/docs/patterns/circuit-breaker.md"
    ),
    "distributed_saga.md": (
        "https://raw.githubusercontent.com/MicrosoftDocs/"
        "architecture-center/main/docs/patterns/saga-content.md"
    ),
    "distributed_idempotent_consumer.md": (
        "https://raw.githubusercontent.com/MicrosoftDocs/"
        "architecture-center/main/docs/patterns/idempotent-consumer.md"
    ),
    "distributed_queue_load_leveling.md": (
        "https://raw.githubusercontent.com/MicrosoftDocs/"
        "architecture-center/main/docs/patterns/queue-based-load-leveling.md"
    ),

    # ==========================================
    # MICROSERVICES — 4 documents
    # ==========================================
    "microservices_gateway_routing.md": (
        "https://raw.githubusercontent.com/MicrosoftDocs/"
        "architecture-center/main/docs/patterns/gateway-routing-content.md"
    ),
    "microservices_gateway_aggregation.md": (
        "https://raw.githubusercontent.com/MicrosoftDocs/"
        "architecture-center/main/docs/patterns/gateway-aggregation.md"
    ),
    "microservices_sidecar.md": (
        "https://raw.githubusercontent.com/MicrosoftDocs/"
        "architecture-center/main/docs/patterns/sidecar.md"
    ),
    "microservices_strangler_fig.md": (
        "https://raw.githubusercontent.com/MicrosoftDocs/"
        "architecture-center/main/docs/patterns/strangler-fig.md"
    ),

    # ==========================================
    # POSTGRESQL — 5 documents
    # ==========================================
    "postgres_indexes.md": (
        "https://raw.githubusercontent.com/supabase/supabase/master/"
        "apps/docs/content/guides/database/postgres/indexes.mdx"
    ),
    "postgres_replication.md": (
        "https://raw.githubusercontent.com/supabase/supabase/master/"
        "apps/docs/content/guides/database/replication.mdx"
    ),
    "postgres_connection_pooling.md": (
        "https://raw.githubusercontent.com/supabase/supabase/master/"
        "apps/docs/content/guides/database/"
        "connecting-to-postgres/pooling-and-limits.mdx"
    ),
    "postgres_connection_management.md": (
        "https://raw.githubusercontent.com/supabase/supabase/master/"
        "apps/docs/content/guides/database/connection-management.mdx"
    ),
    "postgres_index_advisor.md": (
        "https://raw.githubusercontent.com/supabase/supabase/master/"
        "apps/docs/content/guides/database/extensions/index_advisor.mdx"
    ),

    # ==========================================
    # CACHING — 4 documents
    # ==========================================
    "caching_cache_aside.md": (
        "https://raw.githubusercontent.com/MicrosoftDocs/"
        "architecture-center/main/docs/patterns/cache-aside-content.md"
    ),
    "caching_eviction.md": (
        "https://raw.githubusercontent.com/valkey-io/"
        "valkey-doc/main/topics/lru-cache.md"
    ),
    "caching_monitoring.md": (
        "https://raw.githubusercontent.com/valkey-io/"
        "valkey-doc/main/commands/info.md"
    ),
    "caching_pipelining.md": (
        "https://raw.githubusercontent.com/valkey-io/"
        "valkey-doc/main/topics/pipelining.md"
    ),
}



def download_additional_documents():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with httpx.Client(
        timeout=30,
        follow_redirects=True,
    ) as client:
        for filename, url in DOCUMENTS.items():
            target = OUTPUT_DIR / filename

            if target.exists():
                print(f"Skipping: {filename}")
                continue

            try:
                response = client.get(url)
                response.raise_for_status()

                content = response.text

                if len(content.strip()) < 500:
                    raise ValueError("Document too short")

                target.write_text(content, encoding="utf-8")

                print(f"Downloaded: {filename}")

            except (httpx.HTTPError, ValueError) as exc:
                print(f"Failed: {filename}: {exc}")


if __name__ == "__main__":
    download_additional_documents()

    
