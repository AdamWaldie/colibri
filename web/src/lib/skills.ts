/* "/name" at the start of a message turns on the skill called name, when the server
   has one: the rest of the message is what is sent. Anything else is left as typed
   (a path such as "/etc/hosts", or a skill this server does not have). */
export function slashSkill(text: string, known: string[]): { name: string; rest: string } | null {
  const found = /^\/([A-Za-z0-9][A-Za-z0-9._-]{0,63})(?:\s+|$)([\s\S]*)$/.exec(text.trim())
  if (!found || !known.includes(found[1])) return null
  return { name: found[1], rest: found[2].trim() }
}

/* Add a skill once, keeping the order they were turned on in. */
export function withSkill(active: string[], name: string): string[] {
  return active.includes(name) ? active : [...active, name]
}
