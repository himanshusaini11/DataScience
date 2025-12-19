import json, sys, re, glob, os, html

def load_convos():
    if os.path.exists("/Users/enigma/Documents/ChatGPT_Chats/Data/conversations.json"):
        with open("/Users/enigma/Documents/ChatGPT_Chats/Data/conversations.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            # unified shape: make it a list
            return data if isinstance(data, list) else list(data.values())
    files = glob.glob("/Users/enigma/Documents/ChatGPT_Chats/Data/conversations/*.json")
    convos = []
    for p in files:
        with open(p, "r", encoding="utf-8") as f:
            convos.append(json.load(f))
    if not convos:
        sys.exit("No conversations found.")
    return convos

def safe(s): 
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:80]

def flatten_messages(convo):
    # Works with ChatGPT export structure using mapping
    mapping = convo.get("mapping") or {}
    msgs = []
    for node in mapping.values():
        m = node.get("message") or {}
        role = (m.get("author") or {}).get("role")
        content = m.get("content") or {}
        parts = content.get("parts") or []
        text = "\n".join(str(p) for p in parts if isinstance(p, (str,int,float)))
        t = m.get("create_time") or 0
        if role in {"system","user","assistant"} and text.strip():
            msgs.append((t, role, text))
    msgs.sort(key=lambda x: x[0] or 0)
    return msgs

def render_md(convo):
    title = convo.get("title") or "Untitled"
    lines = [f"# {title}", ""]
    for _, role, text in flatten_messages(convo):
        lines.append(f"**{role}**:\n{text}\n")
        lines.append("---")
    return "\n".join(lines[:-1])

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_chat.py '<query or id>'")
        print("Query matches title substring (case-insensitive) or exact id.")
        sys.exit(1)

    q = sys.argv[1].strip().lower()
    convos = load_convos()

    # match by exact id first, else title substring
    hits = [c for c in convos if (c.get("id") or c.get("conversation_id") or "").lower() == q]
    if not hits:
        hits = [c for c in convos if q in (c.get("title") or "").lower()]

    if not hits:
        sys.exit("No chat matched. Try the exact id or a broader title substring.")

    if len(hits) > 1:
        print(f"{len(hits)} matches. Showing titles and ids:")
        for c in hits:
            cid = c.get("id") or c.get("conversation_id") or "unknown-id"
            print(f"- {c.get('title') or 'Untitled'}  |  {cid}")
        sys.exit("Refine your query to a single match.")

    convo = hits[0]
    title = convo.get("title") or "Untitled"
    out = f"chat_{safe(title)}.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(render_md(convo))
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
