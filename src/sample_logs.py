"""
Synthetic log data generator for testing and demonstration.

Generates realistic log events across multiple weeks to demonstrate
trend detection capabilities. Creates three distinct event patterns:

1. Okta authentication failures - grows during weeks 4-8
2. Data access patterns - shifts from S3 to Snowflake after week 6  
3. Vulnerability scans - decays after week 8

The generated data includes realistic timestamps, asset IDs, and contextual
information to test the full pipeline from ingestion to trend detection.
"""
import json, random
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs" / "synthetic.jsonl"

random.seed(42)  # For reproducible synthetic data

def rand_ip() -> str:
    """
    Generate a random IP address for synthetic data.
    
    Returns:
        Random IP address string in dotted decimal notation
    """
    return ".".join(str(random.randint(1, 254)) for _ in range(4))

def main() -> None:
    """
    Generate synthetic log data with predefined trend patterns.
    
    Creates 12 weeks of data with three distinct event types that demonstrate
    different trend patterns: growth, drift, and decay.
    """
    start = datetime(2025, 4, 1, tzinfo=timezone.utc)
    weeks = 12
    rows = []
    
    for w in range(weeks):
        base_ts = start + timedelta(days=7*w)

        # Pattern A: Okta authentication failures (growth trend weeks 4-8)
        nA = 20 + (10 if 4 <= w <= 8 else 0) + random.randint(-3,3)
        for i in range(max(5, nA)):
            ts = base_ts + timedelta(minutes=random.randint(0, 7*24*60))
            rows.append({
                "event_id": f"A-{w}-{i}",
                "ts": ts.isoformat(),
                "product": "okta",
                "event_type": "auth.failure",
                "asset_id": "vpn-1",
                "msg": "MFA denied",
                "context": {"geo": random.choice(["US-CA","US-NY","CN","DE"]), "method": "push"},
                "tech": ["okta","vpn","mfa"],
                "attack": ["TA0006:T1110"],
                "risk_tag": ["identity"]
            })

        # Pattern B: Data access behavior (drift trend - S3 → Snowflake after week 6)
        nB = 25 + random.randint(-5,5)
        for i in range(max(5, nB)):
            ts = base_ts + timedelta(minutes=random.randint(0, 7*24*60))
            if w < 6:
                msg = "S3 GetObject request"; tech = ["aws","s3"]; product = "aws"
            else:
                msg = "Snowflake SELECT large result set"; tech = ["snowflake","sql"]; product = "snowflake"
            rows.append({
                "event_id": f"B-{w}-{i}",
                "ts": ts.isoformat(),
                "product": product,
                "event_type": "data.access",
                "asset_id": random.choice(["etl-1","etl-2","bi-1"]),
                "msg": msg,
                "context": {"region": random.choice(["us-east-1","us-west-2","eu-central-1"])},
                "tech": tech,
                "attack": ["TA0010:T1020"],
                "risk_tag": ["exfiltration"]
            })

        # Pattern C: Vulnerability scan activity (decay trend after week 8)
        nC = (30 if w < 9 else 10) + random.randint(-3,3)
        for i in range(max(5, nC)):
            ts = base_ts + timedelta(minutes=random.randint(0, 7*24*60))
            rows.append({
                "event_id": f"C-{w}-{i}",
                "ts": ts.isoformat(),
                "product": "qualys",
                "event_type": "vuln.finding",
                "asset_id": random.choice(["web-1","db-2","api-3"]),
                "msg": "OpenSSL CVE found",
                "context": {"cvss": random.choice([6.5,7.2,8.1,9.8])},
                "tech": ["openssl","linux"],
                "attack": [],
                "risk_tag": ["vulnerability"]
            })

    # Write all events to JSONL file
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} synthetic events → {OUT}")

if __name__ == "__main__":
    main()
