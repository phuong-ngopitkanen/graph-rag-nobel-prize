from pathlib import Path
import json
from datetime import datetime

data_dir = Path("./data")
data_dir.mkdir(exist_ok=True)

test_data = {
    "total_queries": 2,
    "queries": [
        {
            "query_no": 1,
            "question": "Test question 1",
            "total_time": 5.2
        },
        {
            "query_no": 2,
            "question": "Test question 2",
            "total_time": 3.8
        }
    ]
}

# Write directly - same as teammate
output_path = data_dir / f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
output_path.write_text(json.dumps(test_data, indent=2))

print(f"Test file written to: {output_path}")
print(f"File exists: {output_path.exists()}")
print(f"File size: {output_path.stat().st_size} bytes")