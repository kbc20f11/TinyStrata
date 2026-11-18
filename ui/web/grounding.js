(function(root) {
  "use strict";

  // Use the browser's timezone database. No network request or runtime download.
  function getCurrentTime(args = {}, options = {}) {
    if (!args || typeof args !== "object" || Array.isArray(args) ||
        Object.keys(args).some((key) => key !== "timezone"))
      throw new Error("Expected an optional IANA timezone");
    const zone = args.timezone === undefined
      ? (options.defaultTimeZone || (() => new Intl.DateTimeFormat().resolvedOptions().timeZone))()
      : args.timezone;
    if (typeof zone !== "string" || zone.length > 100 ||
        !/^[A-Za-z_]+(?:\/[A-Za-z0-9_+.\-]+)*$/.test(zone))
      throw new Error("Expected an IANA timezone such as Asia/Tokyo or UTC");
    let formatter;
    try {
      formatter = new Intl.DateTimeFormat("en-US", {
        timeZone: zone, calendar: "gregory", numberingSystem: "latn", hourCycle: "h23",
        year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit",
        minute: "2-digit", second: "2-digit", weekday: "long"
      });
    } catch (_) { throw new Error("Unknown IANA timezone"); }
    // All returned fields describe this one instant, including near midnight/DST.
    const instant = new Date((options.now || (() => Date.now()))());
    if (!Number.isFinite(instant.getTime()) || instant.getUTCFullYear() < 1 || instant.getUTCFullYear() > 9999)
      throw new Error("The computer clock could not be read");
    const parts = Object.fromEntries(formatter.formatToParts(instant)
      .filter((part) => part.type !== "literal").map((part) => [part.type, part.value]));
    const date = `${parts.year.padStart(4, "0")}-${parts.month}-${parts.day}`;
    const local = `${date}T${parts.hour}:${parts.minute}:${parts.second}`;
    // Parsing an explicit Z avoids the host timezone and Date.UTC's 0..99-year rule.
    const offsetSeconds = Math.round((Date.parse(local + "Z") - Math.floor(instant.getTime() / 1000) * 1000) / 1000);
    const absolute = Math.abs(offsetSeconds), pad = (value) => String(value).padStart(2, "0");
    const offset = `${offsetSeconds < 0 ? "-" : "+"}${pad(Math.floor(absolute / 3600))}:${pad(Math.floor(absolute / 60) % 60)}` +
      (absolute % 60 ? `:${pad(absolute % 60)}` : "");
    return {
      source: "computer_clock", timezone: formatter.resolvedOptions().timeZone,
      local_datetime: local + offset, utc_datetime: instant.toISOString(),
      date, weekday: parts.weekday, utc_offset: offset
    };
  }

  const api = {getCurrentTime};
  root.StrataGrounding = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(globalThis);
