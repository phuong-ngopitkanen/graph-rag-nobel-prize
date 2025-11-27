import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

class PerformanceDataExporter:

    def __init__(self, output_dir: str = "./data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.query_log: List[Dict[str, Any]] = []

    def record_query(
        self,
        query_no: int,
        question: str,
        cache_hit: bool,
        total_time: float,
        stage_time: Dict[str, float],
        timestamp: Optional[str] = None
        ) -> None:

        if timestamp is None:
            timestamp = datetime.now().isoformat()

        query_performance_metrics = {
            "query_no": query_no,
            "question": question,
            "cache_hit": cache_hit,
            "total_time": total_time,
            "timestamp": timestamp,
            "stages": stage_time
        }

        self.query_log.append(query_performance_metrics)

    def export_json(
            self,
            profiler_data: Dict[str, Any],
            cache_stats: Dict[str, Any],
            filename: Optional[str] = None

    ) -> Path:
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_metrics_{timestamp}.json"
        
        output_path = self.output_dir / filename

        print(f"\n DEBUG: Attempting to save to: {output_path.absolute()}")
        print(f"DEBUG: Directory exists: {output_path.parent.exists()}")

        data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_queries": len(self.query_log),
                "cache_enabled": cache_stats.get("total_requests", 0) > 0
            },
            "summary": {
                "total_time": profiler_data.get("total_time", 0),
                "cache_hit_rate": cache_stats.get("hit_rate", 0),
                "total_cache_requests": cache_stats.get("total_requests", 0)
            },
            "profiler": profiler_data,
            "cache": cache_stats,
            "queries": self.query_log
        }

        try:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            print(f"Successfully wrote to: {output_path.absolute()}")
        except Exception as e:
            print(f"Failed to write file: {e}")
            raise

        return output_path
    

    def export_csv_summary(
        self,
        profiler_data: Dict[str, Any],
        filename: Optional[str] = None
    ) -> Path:            

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stage_tracker_{timestamp}.csv"

        output_path =  self.output_dir / filename

        stages = profiler_data.get("stages", {})

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "stage_name", "count", "avg_duration", "min_duration", 
                "max_duration", "total_duration", "percentage"
            ])
            
            for stage_name, stage_data in stages.items():
                writer.writerow([
                    stage_name,
                    stage_data.get("count", 0),
                    f"{stage_data.get('avg_duration', 0):.4f}",
                    f"{stage_data.get('min_duration', 0):.4f}",
                    f"{stage_data.get('max_duration', 0):.4f}",
                    f"{stage_data.get('total_duration', 0):.4f}",
                    f"{stage_data.get('percentage', 0):.2f}"
                ])       
            
        print(f"stage time tracker csv exported to: {output_path}")
        return output_path
    
    def export_csv_query_log(self, filename: Optional[str] = None) -> Path:

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"query_log_{timestamp}.csv"

        output_path = self.output_dir / filename
        
        if not self.query_log:
            print("No query log data to export")
            return output_path

        all_stages = set()
        for query in self.query_log:
            all_stages.update(query.get("stages", {}).keys())
        
        stage_columns = sorted(all_stages)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ["query_no", "question", "cache_hit", "total_time", "timestamp"] + stage_columns
            writer.writerow(header)
            
            for query in self.query_log:
                row = [
                    query["query_no"],
                    query["question"][:50] + "..." if len(query["question"]) > 50 else query["question"],
                    query["cache_hit"],
                    f"{query['total_time']:.4f}",
                    query["timestamp"]
                ]
                
                for stage in stage_columns:
                    stage_time = query.get("stages", {}).get(stage, 0)
                    row.append(f"{stage_time:.4f}")
                
                writer.writerow(row)
        print(f"Query log CSV exported to: {output_path}")
        return output_path
    
    def export_simple(self, filename: str = None) -> Path:
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_{timestamp}.json"
    
        if not filename.endswith('.json'):
            filename = f"{filename}.json"
    
        output_path = self.output_dir / filename
    
        data = {
            "total_queries": len(self.query_log),
            "queries": self.query_log
    }
    
        output_path.write_text(json.dumps(data, indent=2, default=str))
    
        print(f"Data exported to: {output_path}")
        return output_path

    def export_all(
        self,
        profiler_data: Dict[str, Any],
        cache_stats: Dict[str, Any],
        base_filename: Optional[str] = None
    ) -> Dict[str, Path]:
  
        if base_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"performance_metrics_{timestamp}"
        
        paths = {
            "json": self.export_json(
                profiler_data, 
                cache_stats, 
                f"{base_filename}.json"
            ),
            "stage_csv": self.export_csv_summary(
                profiler_data, 
                f"{base_filename}_stages.csv"
            ),
            "query_csv": self.export_csv_query_log(
                f"{base_filename}_queries.csv"
            )
        }
        
        print(f"\n{'='*60}")
        print("All performance data exported successfully")
        print(f"{'='*60}")
        return paths
    
    def clear_log(self) -> None:
        self.query_log.clear()

def load_performance_data(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r') as f:
        return json.load(f)


def quick_export(
    profiler,
    cache,
    query_log: Optional[List[Dict]] = None,
    output_dir: str = "./data"
) -> Dict[str, Path]:

    exporter = PerformanceDataExporter(output_dir)
    

    if query_log:
        exporter.query_log = query_log
    
    profiler_data = profiler.get_summary_dict()
    cache_stats = {
        "total_requests": cache.stats.total_requests,
        "hits": cache.stats.hits,
        "misses": cache.stats.misses,
        "hit_rate": cache.stats.hit_rate,
        "evictions": cache.stats.evictions,
        "cache_size": len(cache.cache)
    }
    
    return exporter.export_all(profiler_data, cache_stats)







        