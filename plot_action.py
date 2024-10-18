import json
from collections import Counter
import matplotlib.pyplot as plt

with open('nusc_action_train.json', 'r') as f:
    data = json.load(f)


commands = []
for scene in data.values():
    for command in scene.values():
        commands.extend(command.split(', '))


command_counts = Counter(commands)


labels, values = zip(*command_counts.items())
plt.figure(figsize=(10, 6))
plt.bar(labels, values, color='skyblue')

plt.title('Command Distribution')
plt.xlabel('Commands')
plt.ylabel('Frequency')


plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("cmd_train.jpg", dpi=300)
# plt.show()
