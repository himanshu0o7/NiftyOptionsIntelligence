# Tests for NiftyOptionsIntelligence

This directory contains comprehensive automated tests for the `greeks_handler.py` module.

## Test Coverage

### Unit Tests (`TestGetOptionGreeksDummy`)
- Tests for the dummy `get_option_greeks` function
- Validates input parameter validation 
- Tests edge cases with invalid symbols, strikes, and option types
- Verifies correct return format for valid inputs

### Integration Tests (`TestFetchOptionGreeksUnit`)
- Tests for the main `fetch_option_greeks` function with mocked dependencies
- Mocks external dependencies (SmartAPI, WebSocket, network calls)
- Tests success scenarios and error conditions
- Validates proper handling of API responses

### End-to-End Tests (`TestGreeksHandlerIntegration`)
- Tests interaction between different components
- Validates global variable persistence
- Tests WebSocket data flow simulation
- Multi-symbol processing validation

### Edge Case Tests (`TestGreeksHandlerEdgeCases`)
- Boundary value testing
- Thread safety simulation
- Symbol format variations
- Error propagation testing

## Running Tests

### Run all tests:
```bash
python -m unittest tests.test_greeks_handler -v
```

### Run specific test class:
```bash
python -m unittest tests.test_greeks_handler.TestGetOptionGreeksDummy -v
```

### Run with the test runner:
```bash
python tests/run_tests.py
```

## Test Structure

- **24 total test cases** covering all major functionality
- **Mocked external dependencies** to ensure tests are isolated and fast
- **Edge case coverage** for robust error handling
- **Integration tests** to verify component interaction

## Dependencies

The tests require the following packages:
- `unittest` (Python standard library)
- `pandas`
- `smartapi-python`
- `logzero`
- `websocket-client`

All external API calls are mocked to ensure tests are reliable and don't depend on network connectivity.