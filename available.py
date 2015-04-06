import re

start = re.compile('(<h2>Available Rooms</h2>)');
pattern = re.compile('(<b>([^< ]*)[^<]*</b>)')
end = re.compile('(</div>)');

def get_available_rooms():
    content = ""
    with open('text.html', 'r') as f:
        for line in f:
            content += line
    content = content[start.search(content).end(1):]
    content = content[:end.search(content).start(1)]
    available = set()
    while True:
        match = pattern.search(content)
        if match is None:
            break
        rm_num = match.group(2)
        content = content[match.end(1):]
        available.add(rm_num)
    return available
