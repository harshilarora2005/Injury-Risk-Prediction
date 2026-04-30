/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        // Matches timeline.png palette
        bg:       "#0f1117",
        panel:    "#1a1d27",
        panel2:   "#222633",
        border:   "#2a2f3d",
        muted:    "#8a90a2",
        ink:      "#e6e8ef",
        risk: {
          low:    "#00c853",
          med:    "#ffd600",
          high:   "#ff1744",
          highBg: "#ff174422",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(255,23,68,.25), 0 8px 30px -12px rgba(255,23,68,.45)",
      },
    },
  },
  plugins: [],
};
