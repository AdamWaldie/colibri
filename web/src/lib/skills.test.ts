import { describe, expect, it } from "vitest"

import { slashSkill, withSkill } from "./skills"

describe("skills", () => {
  const known = ["code-review", "pdf"]

  it("turns a skill on from /name at the start of a message", () => {
    expect(slashSkill("/code-review look at this diff", known)).toEqual({ name: "code-review", rest: "look at this diff" })
    expect(slashSkill("  /pdf\nfill it in", known)).toEqual({ name: "pdf", rest: "fill it in" })
    expect(slashSkill("/pdf", known)).toEqual({ name: "pdf", rest: "" })
  })

  it("leaves anything else as typed", () => {
    expect(slashSkill("/etc/hosts is odd", known)).toBeNull()
    expect(slashSkill("/unknown do it", known)).toBeNull()
    expect(slashSkill("use /pdf here", known)).toBeNull()
  })

  it("adds a skill once", () => {
    expect(withSkill(["pdf"], "pdf")).toEqual(["pdf"])
    expect(withSkill(["pdf"], "code-review")).toEqual(["pdf", "code-review"])
  })
})
