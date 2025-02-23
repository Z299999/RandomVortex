class State:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None # tail state is the current frame we want to animate
        self.step = 0 # delta t, the small time interval

    def evolve(self, data):
        new_state = State(data)
        if not self.head:
            self.head = new_state
            self.tail = new_state
            return 
        last_state = self.head
        while last_state:
            last_state = last_state.next
        last_state.next = new_state
        self.tail = new_state

ll = LinkedList()
print(ll.head)
print(ll.tail)
ll.evolve(1)
print(ll.head)
print(ll.tail.data)

        
