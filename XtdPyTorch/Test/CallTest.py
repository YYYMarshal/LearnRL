class Person:
    def __call__(self, name):
        print(f"call: {name}")

    def hello(self, name):
        print(f"hello: {name}")


person = Person()
person("AAA")
person.hello("BBB")
