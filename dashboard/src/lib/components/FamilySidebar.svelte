<script lang="ts">
  import FamilyLogos from "./FamilyLogos.svelte";

  type FamilySidebarProps = {
    families: string[];
    selectedFamily: string | null;
    hasFavorites: boolean;
    hasRecents: boolean;
    hasShared?: boolean;
    sharedDirectory?: string | null;
    onSelect: (family: string | null) => void;
  };

  let {
    families,
    selectedFamily,
    hasFavorites,
    hasRecents,
    hasShared = false,
    sharedDirectory = null,
    onSelect,
  }: FamilySidebarProps = $props();

  // Family display names
  const familyNames: Record<string, string> = {
    favorites: "Favorites",
    recents: "Recent",
    huggingface: "Hub",
    llama: "Meta",
    qwen: "Qwen",
    deepseek: "DeepSeek",
    "gpt-oss": "OpenAI",
    glm: "GLM",
    minimax: "MiniMax",
    kimi: "Kimi",
    flux: "FLUX",
    "qwen-image": "Qwen Img",
    nemotron: "NVIDIA",
    gemma: "Google",
  };

  function getFamilyName(family: string): string {
    return (
      familyNames[family] || family.charAt(0).toUpperCase() + family.slice(1)
    );
  }
</script>

<div
  class="flex flex-col gap-1 py-2 px-1 border-r border-xeo-green/10 bg-xeo-medium-gray/30 min-w-[80px] sm:min-w-[72px] overflow-y-auto scrollbar-hide"
>
  <!-- All models (no filter) -->
  <button
    type="button"
    onclick={() => onSelect(null)}
    class="group flex items-center justify-center px-3 py-2.5 rounded transition-all duration-200 cursor-pointer min-h-[44px] sm:min-h-0 {selectedFamily ===
    null
      ? 'bg-xeo-green/20 border-l-2 border-xeo-green'
      : 'hover:bg-white/5 border-l-2 border-transparent'}"
    title="All models"
  >
    <span
      class="text-[12px] font-mono font-medium {selectedFamily === null
        ? 'text-xeo-green'
        : 'text-white/40 group-hover:text-white/60'}">All</span
    >
  </button>

  <!-- Models already available on at least one node -->
  <button
    type="button"
    onclick={() => onSelect("downloaded")}
    class="group flex flex-col items-center justify-center p-2 rounded transition-all duration-200 cursor-pointer {selectedFamily ===
    'downloaded'
      ? 'bg-xeo-green/20 border-l-2 border-xeo-green'
      : 'hover:bg-white/5 border-l-2 border-transparent'}"
    title="Show downloaded models"
  >
    <svg
      class="w-5 h-5 {selectedFamily === 'downloaded'
        ? 'text-xeo-green'
        : 'text-white/50 group-hover:text-xeo-green/70'}"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="1.6"
      stroke-linecap="round"
      stroke-linejoin="round"
      aria-hidden="true"
    >
      <path d="M12 3v11" />
      <path d="m8 10 4 4 4-4" />
      <path d="M5 18h14" />
    </svg>
    <span
      class="text-[11px] font-mono mt-0.5 {selectedFamily === 'downloaded'
        ? 'text-xeo-green'
        : 'text-white/40 group-hover:text-white/60'}">Saved</span
    >
  </button>

  <!-- Shared folder (when Shared Model Storage is configured) -->
  {#if hasShared}
    <button
      type="button"
      onclick={() => onSelect("shared")}
      class="group flex flex-col items-center justify-center p-2 rounded transition-all duration-200 cursor-pointer {selectedFamily ===
      'shared'
        ? 'bg-cyan-400/20 border-l-2 border-cyan-400'
        : 'hover:bg-white/5 border-l-2 border-transparent'}"
      title={sharedDirectory
        ? `Shared model storage — ${sharedDirectory}`
        : "Shared model storage"}
    >
      <svg
        class="w-5 h-5 {selectedFamily === 'shared'
          ? 'text-cyan-300'
          : 'text-white/50 group-hover:text-cyan-300/70'}"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="1.6"
        stroke-linecap="round"
        stroke-linejoin="round"
        aria-hidden="true"
      >
        <path d="M4 5h16v6H4zM4 15h16v4H4z" />
        <path d="M8 8h.01M8 17h.01" />
      </svg>
      <span
        class="text-[11px] font-mono mt-0.5 {selectedFamily === 'shared'
          ? 'text-cyan-300'
          : 'text-white/40 group-hover:text-white/60'}">Shared</span
      >
    </button>
  {/if}

  <!-- Favorites (only show if has favorites) -->
  {#if hasFavorites}
    <button
      type="button"
      onclick={() => onSelect("favorites")}
      class="group flex flex-col items-center justify-center p-2 rounded transition-all duration-200 cursor-pointer {selectedFamily ===
      'favorites'
        ? 'bg-xeo-green/20 border-l-2 border-xeo-green'
        : 'hover:bg-white/5 border-l-2 border-transparent'}"
      title="Show favorited models"
    >
      <FamilyLogos
        family="favorites"
        class={selectedFamily === "favorites"
          ? "text-amber-400"
          : "text-white/50 group-hover:text-amber-400/70"}
      />
      <span
        class="text-[11px] font-mono mt-0.5 {selectedFamily === 'favorites'
          ? 'text-amber-400'
          : 'text-white/40 group-hover:text-white/60'}">Faves</span
      >
    </button>
  {/if}

  <!-- Recent (only show if has recent models) -->
  {#if hasRecents}
    <button
      type="button"
      onclick={() => onSelect("recents")}
      class="group flex flex-col items-center justify-center p-2 rounded transition-all duration-200 cursor-pointer {selectedFamily ===
      'recents'
        ? 'bg-xeo-green/20 border-l-2 border-xeo-green'
        : 'hover:bg-white/5 border-l-2 border-transparent'}"
      title="Recently launched models"
    >
      <FamilyLogos
        family="recents"
        class={selectedFamily === "recents"
          ? "text-xeo-green"
          : "text-white/50 group-hover:text-white/70"}
      />
      <span
        class="text-[11px] font-mono mt-0.5 {selectedFamily === 'recents'
          ? 'text-xeo-green'
          : 'text-white/40 group-hover:text-white/60'}">Recent</span
      >
    </button>
  {/if}

  <!-- HuggingFace Hub -->
  <button
    type="button"
    onclick={() => onSelect("huggingface")}
    class="group flex flex-col items-center justify-center p-2 rounded transition-all duration-200 cursor-pointer {selectedFamily ===
    'huggingface'
      ? 'bg-orange-500/20 border-l-2 border-orange-400'
      : 'hover:bg-white/5 border-l-2 border-transparent'}"
    title="Browse and add models from Hugging Face"
  >
    <FamilyLogos
      family="huggingface"
      class={selectedFamily === "huggingface"
        ? "text-orange-400"
        : "text-white/50 group-hover:text-orange-400/70"}
    />
    <span
      class="text-[11px] font-mono mt-0.5 {selectedFamily === 'huggingface'
        ? 'text-orange-400'
        : 'text-white/40 group-hover:text-white/60'}">Hub</span
    >
  </button>

  <div class="h-px bg-xeo-green/10 my-1"></div>

  <!-- Model families -->
  {#each families as family}
    <button
      type="button"
      onclick={() => onSelect(family)}
      class="group flex flex-col items-center justify-center p-2 rounded transition-all duration-200 cursor-pointer {selectedFamily ===
      family
        ? 'bg-xeo-green/20 border-l-2 border-xeo-green'
        : 'hover:bg-white/5 border-l-2 border-transparent'}"
      title={getFamilyName(family)}
    >
      <FamilyLogos
        {family}
        class={selectedFamily === family
          ? "text-xeo-green"
          : "text-white/50 group-hover:text-white/70"}
      />
      <span
        class="text-[11px] font-mono mt-0.5 truncate max-w-full {selectedFamily ===
        family
          ? 'text-xeo-green'
          : 'text-white/40 group-hover:text-white/60'}"
      >
        {getFamilyName(family)}
      </span>
    </button>
  {/each}
</div>
