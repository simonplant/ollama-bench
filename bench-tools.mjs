// Shared tool catalogue for bench-toolcall.mjs and bench-multiturn.mjs.
// LifeOps-shaped: email, calendar, tasks, quote, web search.

export const TOOLS = [
  {
    type: "function", function: {
      name: "email_inbox",
      description: "List unread emails in the inbox. Returns id, subject, from, date.",
      parameters: { type: "object", properties: {
        limit: { type: "integer", description: "Max emails to return", default: 20 },
        folder: { type: "string", description: "Folder name", default: "INBOX" }
      }, required: [] },
    },
  },
  {
    type: "function", function: {
      name: "email_read",
      description: "Read the body of an email by id.",
      parameters: { type: "object", properties: {
        id: { type: "string", description: "Email id" }
      }, required: ["id"] },
    },
  },
  {
    type: "function", function: {
      name: "email_send",
      description: "Send an email.",
      parameters: { type: "object", properties: {
        to:      { type: "string", description: "Recipient email address" },
        subject: { type: "string" },
        body:    { type: "string" },
      }, required: ["to", "subject", "body"] },
    },
  },
  {
    type: "function", function: {
      name: "calendar_events",
      description: "List calendar events in a date range.",
      parameters: { type: "object", properties: {
        start: { type: "string", description: "ISO date (YYYY-MM-DD)" },
        end:   { type: "string", description: "ISO date (YYYY-MM-DD)" },
      }, required: ["start", "end"] },
    },
  },
  {
    type: "function", function: {
      name: "task_create",
      description: "Create a task in the to-do list.",
      parameters: { type: "object", properties: {
        title:    { type: "string" },
        due:      { type: "string", description: "ISO date or natural language" },
        priority: { type: "string", enum: ["low", "medium", "high"] },
      }, required: ["title"] },
    },
  },
  {
    type: "function", function: {
      name: "quote",
      description: "Get a real-time stock quote.",
      parameters: { type: "object", properties: {
        symbol: { type: "string", description: "Ticker symbol, e.g. AAPL" }
      }, required: ["symbol"] },
    },
  },
  {
    type: "function", function: {
      name: "web_search",
      description: "Search the web.",
      parameters: { type: "object", properties: {
        query: { type: "string" },
        limit: { type: "integer", default: 5 }
      }, required: ["query"] },
    },
  },
];
