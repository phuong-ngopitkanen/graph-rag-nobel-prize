import sys
from profiler import Profiler
import time


def test_profiler_basic():
    print("=" * 80)
    print("Test 1: Basic Profiler Functionality")
    print("=" * 80)
    
    profiler = Profiler()
    
    with profiler.profile("stage1"):
        time.sleep(0.1)  
    
    profiler.start("stage2")
    time.sleep(0.05)  
    profiler.stop("stage2")
    
    for i in range(3):
        with profiler.profile("repeated_stage"):
            time.sleep(0.02)  
    
    
    profiler.print_summary()

    results = profiler.get_results()
    assert "stage1" in results
    assert "stage2" in results
    assert "repeated_stage" in results
    assert results["repeated_stage"].count == 3
    
    print("Basic profiler tests passed!\n")


def test_profiler_nested():
    print("=" * 80)
    print("Test 2: Nested profiling")
    print("=" * 80)
    
    profiler = Profiler()
    
    with profiler.profile("outer_stage"):
        time.sleep(0.05)
        
        with profiler.profile("inner_stage_1"):
            time.sleep(0.1)
        
        with profiler.profile("inner_stage_2"):
            time.sleep(0.05)
    
    profiler.print_summary()
    
    results = profiler.get_results()
    outer_time = results["outer_stage"].total_duration
    inner1_time = results["inner_stage_1"].total_duration
    inner2_time = results["inner_stage_2"].total_duration
    
    assert outer_time >= (inner1_time + inner2_time)
    
    print("Nested profiling tests passed\n")


def test_profiler_statistics():
    print("=" * 80)
    print("Test 3: Statistics calculation")
    print("=" * 80)
    
    profiler = Profiler()
    
    durations = [0.1, 0.2, 0.15, 0.25, 0.12]
    for duration in durations:
        with profiler.profile("variable_stage"):
            time.sleep(duration)
    
    profiler.print_summary()
    
    results = profiler.get_results()
    stage = results["variable_stage"]
    
    assert stage.count == 5
    assert abs(stage.min_duration - 0.1) < 0.05  
    assert abs(stage.max_duration - 0.25) < 0.05
    assert abs(stage.avg_duration - 0.164) < 0.05
    
    print("Statistics tests passed\n")


def test_profiler_reset():
    print("=" * 80)
    print("Test 4: Reset functionality")
    print("=" * 80)
    
    profiler = Profiler()
    
    with profiler.profile("stage1"):
        time.sleep(0.05)
    
    print("Before reset:")
    profiler.print_summary()
    
    # Reset
    profiler.reset()
    
    print("\nAfter reset:")
    profiler.print_summary()
    
    results = profiler.get_results()
    assert len(results) == 0
    
    print("Reset tests passed\n")


def test_format_duration():
    print("=" * 80)
    print("Test 5: Duration formatting")
    print("=" * 80)
    
    from profiler import format_duration
    
    tests = [
        (2.5, "2.50s"),
        (0.123, "123.00ms"),
        (0.001, "1.00ms"),
        (10.456, "10.46s"),
    ]
    
    for duration, expected in tests:
        result = format_duration(duration)
        print(f"  {duration}s -> {result} (expected: {expected})")
        assert result == expected, f"Expected {expected}, got {result}"
    
    print("Duration formatting tests passed\n")


def run_all_tests():
    print("\n")
    print("*" * 80)
    print("Profiler test")
    print("*" * 80)
    print("\n")
    
    try:
        test_profiler_basic()
        test_profiler_nested()
        test_profiler_statistics()
        test_profiler_reset()
        test_format_duration()
        
        print("\n")
        print("*" * 80)
        print("All tests passed")
        print("*" * 80)
        print("\n")
        print("Your profiler is working correctly")
        
    except AssertionError as e:
        print(f"\n Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()






