from cocoa.core.kb import KB as BaseKB

class KB(BaseKB):
    def __init__(self, attributes, items):
        super(KB, self).__init__(attributes)
        self.items = []
        for name, value in items['Item_values'].items():
            self.items.append({'Name': name, 'Value': value,
                               'Count': items['Item_counts'][name]})
        self.item_counts = {item['Name']: item['Count'] for item in self.items}
        self.item_values = {item['Name']: item['Value'] for item in self.items}

    def to_dict(self):
        return self.items

    @classmethod
    def from_dict(cls, attributes, raw):
        return cls(attributes, raw)

    @classmethod
    def from_ints(cls, attributes, names, ints):
        """Build KB from integers.

        Args:
            names (list[str])
            ints (list[int]): [count1, value1, count2, value2, ...]

        """
        items = []
        assert 1. * len(ints) / len(names) == 2
        for i, name in enumerate(names):
            item = {'Name': name, 'Count': ints[i*2], 'Value': ints[i*2+1]}
            items.append(item)
        return cls(attributes, items)

    def dump(self):
        item_counts = ', '.join(['{count} {item}s'.format(count=c, item=n) for n, c in self.item_counts.items()])
        print('Items Available: {}'.format(item_counts))

        for item, value in self.item_values.items():
            print('How you value {0}: {1} points'.format(item, value))
        print('----------------')
