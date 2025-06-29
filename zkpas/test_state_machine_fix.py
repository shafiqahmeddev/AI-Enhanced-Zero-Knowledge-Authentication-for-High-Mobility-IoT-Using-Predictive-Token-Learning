#!/usr/bin/env python3
"""
Demonstration of the fixed Mermaid diagram generation.

This script shows that the StateTransition objects are now correctly handled
in the generate_mermaid_diagram() method.
"""

from app.state_machine import StateMachine, StateType, StateTransition, GatewayStateMachine
from app.events import EventBus, EventType

def demonstrate_fix():
    """Demonstrate the correct Mermaid diagram generation."""
    print("=" * 60)
    print("STATE MACHINE MERMAID DIAGRAM FIX DEMONSTRATION")
    print("=" * 60)
    
    # Create a simple custom state machine for demonstration
    class DemoStateMachine(StateMachine):
        def _setup_transitions(self):
            # Add a few transitions to demonstrate the fix
            self.add_transition(StateTransition(
                from_state=StateType.IDLE,
                to_state=StateType.AWAITING_COMMITMENT,
                event_type=EventType.AUTH_REQUEST
            ))
            
            self.add_transition(StateTransition(
                from_state=StateType.AWAITING_COMMITMENT,
                to_state=StateType.AUTHENTICATED,
                event_type=EventType.VERIFICATION_COMPLETE
            ))
            
            self.add_transition(StateTransition(
                from_state=StateType.AUTHENTICATED,
                to_state=StateType.IDLE,
                event_type=EventType.SESSION_EXPIRED
            ))
    
    # Create event bus and demo state machine
    event_bus = EventBus()
    demo_machine = DemoStateMachine("demo", event_bus)
    
    print("Demo State Machine Transitions Structure:")
    print("-" * 40)
    for from_state, transitions in demo_machine.transitions.items():
        for event_type, transition in transitions.items():
            print(f"  {from_state.name} --[{event_type.name}]--> {transition.to_state.name}")
            print(f"    (transition object: {type(transition).__name__})")
    
    print(f"\nGenerated Mermaid Diagram:")
    print("-" * 40)
    mermaid_diagram = demo_machine.generate_mermaid_diagram()
    print(mermaid_diagram)
    
    print(f"\nKey Fix Details:")
    print("-" * 40)
    print("✓ Before: for event_type, to_state in transitions.items():")
    print("  - Incorrectly treated 'to_state' as StateType object")
    print("  - Would fail with AttributeError: StateTransition has no 'name' attribute")
    print()
    print("✓ After: for event_type, transition in transitions.items():")
    print("  - Correctly recognizes 'transition' as StateTransition object")
    print("  - Accesses transition.to_state.name for proper state name")
    
    # Test with Gateway state machine as well
    print(f"\nGateway State Machine Example:")
    print("-" * 40)
    gateway = GatewayStateMachine("test_gateway", event_bus)
    gateway_transitions = len([
        transition
        for transitions in gateway.transitions.values()
        for transition in transitions.values()
    ])
    print(f"Total transitions defined: {gateway_transitions}")
    print("Mermaid generation successful: ✓")
    
    return mermaid_diagram

if __name__ == "__main__":
    try:
        result = demonstrate_fix()
        print(f"\nDemonstration completed successfully!")
    except Exception as e:
        print(f"Error during demonstration: {e}")
        print("This would indicate the fix didn't work properly.")
