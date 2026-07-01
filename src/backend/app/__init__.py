"""App bridge layer — orchestrates game creation and AI card selection.

This package bridges the controller/, game/, and ai/ packages.
Per design/3 architecture:
    - AppManager: game creation, AI action selection, model lifecycle
    - Converter: one-hot ↔ card conversion, input vector construction
"""
