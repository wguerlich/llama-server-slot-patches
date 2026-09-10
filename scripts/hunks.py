#!/usr/bin/env python3
"""Split a unified diff into hunks, select a subset, write it back out.

  hunks.py list  <patch>                 -> every hunk with file, number, target line, first line
  hunks.py cut   <patch> <spec> <out>    -> write only the hunks NOT excluded by spec
        spec: "file:1,3-5;other:2"  (hunk numbers to exclude, per file)
"""
import sys, re

def parse(text):
    files = []   # [{hdr, hunks, path}]
    cur = None
    for line in text.split('\n'):
        if line.startswith('--- a/'):
            cur = {'hdr': [line], 'hunks': [], 'path': line[6:].strip()}
            files.append(cur)
        elif cur is not None and line.startswith('+++ ') and len(cur['hdr']) == 1:
            cur['hdr'].append(line)
        elif cur is not None and line.startswith('@@'):
            cur['hunks'].append([line])
        elif cur is not None and cur['hunks']:
            cur['hunks'][-1].append(line)
    return files

def main():
    if len(sys.argv) < 3:
        print(__doc__); return 1
    mode, path = sys.argv[1], sys.argv[2]
    files = parse(open(path).read())
    if mode == 'list':
        for f in files:
            for i, h in enumerate(f['hunks'], 1):
                m = re.match(r'@@ -(\d+)', h[0])
                first = next((l for l in h[1:] if l[:1] in '+-'), '')
                print("%-38s #%-2d @%-6s %s" % (f['path'], i, m.group(1) if m else '?', first[:78]))
        return 0
    if mode == 'cut':
        spec, out = sys.argv[3], sys.argv[4]
        drop = {}
        for part in spec.split(';'):
            if not part.strip(): continue
            fn, nums = part.split(':')
            s = set()
            for tok in nums.split(','):
                if '-' in tok:
                    a, b = tok.split('-'); s.update(range(int(a), int(b)+1))
                elif tok.strip():
                    s.add(int(tok))
            drop[fn.strip()] = s
        o = []
        for f in files:
            keep = [h for i, h in enumerate(f['hunks'], 1) if i not in drop.get(f['path'], set())]
            if not keep: continue
            o.extend(f['hdr'])
            for h in keep: o.extend(h)
        open(out, 'w').write('\n'.join(l for l in o if l is not None).rstrip('\n') + '\n')
        print("  wrote: %s (%d lines)" % (out, len(o)))
        return 0
    print(__doc__); return 1

sys.exit(main())
