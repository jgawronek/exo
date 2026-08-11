<script lang="ts">
  import { browser } from "$app/environment";
  import { featureFlags, topologyData } from "$lib/stores/app.svelte";

  const showAdvanced = $derived(featureFlags()["disaggregation"] === true);
  const topology = $derived(topologyData());

  /** Cluster-wide memory: available first, then total (same source as model picker). */
  const clusterMemory = $derived.by(() => {
    const nodes = topology?.nodes;
    if (!nodes) return null;
    let totalBytes = 0;
    let usedBytes = 0;
    for (const node of Object.values(nodes)) {
      const total = node.macmon_info?.memory?.ram_total ?? 0;
      const used = node.macmon_info?.memory?.ram_usage ?? 0;
      if (total <= 0) continue;
      totalBytes += total;
      usedBytes += used;
    }
    if (totalBytes <= 0) return null;
    return {
      availableBytes: Math.max(totalBytes - usedBytes, 0),
      totalBytes,
    };
  });

  interface Props {
    showHome?: boolean;
    onHome?: (() => void) | null;
    showSidebarToggle?: boolean;
    sidebarVisible?: boolean;
    onToggleSidebar?: (() => void) | null;
    showMobileMenuToggle?: boolean;
    mobileMenuOpen?: boolean;
    onToggleMobileMenu?: (() => void) | null;
    showMobileRightToggle?: boolean;
    mobileRightOpen?: boolean;
    onToggleMobileRight?: (() => void) | null;
    downloadProgress?: {
      count: number;
      percentage: number;
    } | null;
  }

  let {
    showHome = true,
    onHome = null,
    showSidebarToggle = false,
    sidebarVisible = true,
    onToggleSidebar = null,
    showMobileMenuToggle = false,
    mobileMenuOpen = false,
    onToggleMobileMenu = null,
    showMobileRightToggle = false,
    mobileRightOpen = false,
    onToggleMobileRight = null,
    downloadProgress = null,
  }: Props = $props();

  function formatMemoryGb(bytes: number): string {
    return `${Math.round(bytes / (1024 * 1024 * 1024))}GB`;
  }

  function handleHome(): void {
    if (onHome) {
      onHome();
      return;
    }
    if (browser) {
      // Hash router: send to root
      window.location.hash = "/";
    }
  }

  function handleToggleSidebar(): void {
    if (onToggleSidebar) {
      onToggleSidebar();
    }
  }

  function handleToggleMobileMenu(): void {
    if (onToggleMobileMenu) {
      onToggleMobileMenu();
    }
  }

  function handleToggleMobileRight(): void {
    if (onToggleMobileRight) {
      onToggleMobileRight();
    }
  }
</script>

<header
  class="relative z-20 flex items-center px-4 md:px-6 pt-4 md:pt-8 pb-3 md:pb-4 bg-xeo-dark-gray"
>
  <!-- Left: Sidebar Toggle, positioned after the persistent wordmark -->
  <div class="order-2 ml-2 md:ml-3 flex items-center gap-2">
    <!-- Mobile sidebar toggle -->
    <button
      onclick={handleToggleMobileMenu}
      class="p-2 rounded border border-xeo-light-gray/30 hover:border-xeo-green/50 hover:bg-xeo-medium-gray/30 transition-colors cursor-pointer md:hidden"
      title={mobileMenuOpen ? "Hide sidebar" : "Show sidebar"}
      aria-label={mobileMenuOpen
        ? "Hide conversation sidebar"
        : "Show conversation sidebar"}
      aria-pressed={mobileMenuOpen}
    >
      <svg
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        stroke-width="2"
        class="w-5 h-5 {mobileMenuOpen
          ? 'text-xeo-green'
          : 'text-xeo-light-gray'}"
      >
        {#if mobileMenuOpen}
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            d="M11 19l-7-7 7-7m8 14l-7-7 7-7"
          ></path>
        {:else}
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            d="M13 5l7 7-7 7M5 5l7 7-7 7"
          ></path>
        {/if}
      </svg>
    </button>
    <!-- Desktop sidebar toggle -->
    <button
      onclick={handleToggleSidebar}
      class="p-2 rounded border border-xeo-light-gray/30 hover:border-xeo-green/50 hover:bg-xeo-medium-gray/30 transition-colors cursor-pointer hidden md:block"
      title={sidebarVisible ? "Hide sidebar" : "Show sidebar"}
      aria-label={sidebarVisible
        ? "Hide conversation sidebar"
        : "Show conversation sidebar"}
      aria-pressed={sidebarVisible}
    >
      <svg
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        stroke-width="2"
        class="w-5 h-5 {sidebarVisible
          ? 'text-xeo-green'
          : 'text-xeo-light-gray'}"
      >
        {#if sidebarVisible}
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            d="M11 19l-7-7 7-7m8 14l-7-7 7-7"
          ></path>
        {:else}
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            d="M13 5l7 7-7 7M5 5l7 7-7 7"
          ></path>
        {/if}
      </svg>
    </button>
  </div>

  <!-- Persistent top-left wordmark (clickable to go home) -->
  <button
    onclick={handleHome}
    class="order-1 flex flex-col items-center gap-0.5 bg-transparent border-none outline-none focus:outline-none transition-opacity duration-200 hover:opacity-90 {showHome
      ? 'cursor-pointer'
      : 'cursor-default'}"
    title={showHome ? "Go to home" : ""}
    disabled={!showHome}
  >
    <img
      src="/xeo-wordmark.svg"
      alt="XEO"
      class="h-7 md:h-10 drop-shadow-[0_0_4px_oklch(0.78_0.17_145/0.3)]"
    />
    <span
      class="text-[9px] md:text-[10px] font-mono tracking-[0.28em] uppercase text-xeo-green/70 leading-none"
    >
      Beta
    </span>
  </button>

  <!-- Right: Library / Server + cluster memory -->
  <nav
    class="absolute right-4 md:right-6 top-1/2 -translate-y-1/2 flex flex-col items-end gap-1"
    aria-label="Main navigation"
  >
    <div class="flex items-center gap-2 md:gap-4">
      <!-- Mobile right sidebar toggle (instances/models) - only show when not in chat mode -->
      {#if showMobileRightToggle}
        <button
          onclick={handleToggleMobileRight}
          class="p-2 rounded border border-xeo-light-gray/30 hover:border-xeo-green/50 hover:bg-xeo-medium-gray/30 transition-colors cursor-pointer md:hidden"
          title={mobileRightOpen ? "Hide instances" : "Show instances"}
          aria-label={mobileRightOpen
            ? "Hide instances panel"
            : "Show instances panel"}
          aria-pressed={mobileRightOpen}
        >
          <svg
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            stroke-width="2"
            class="w-5 h-5 {mobileRightOpen
              ? 'text-xeo-green'
              : 'text-xeo-light-gray'}"
          >
            {#if mobileRightOpen}
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                d="M13 5l7 7-7 7M5 5l7 7-7 7"
              ></path>
            {:else}
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                d="M11 19l-7-7 7-7m8 14l-7-7 7-7"
              ></path>
            {/if}
          </svg>
        </button>
      {/if}
      <a
        href="/#/library"
        class="text-xs md:text-sm text-white/70 hover:text-xeo-green transition-colors tracking-wider uppercase flex items-center gap-1.5 md:gap-2 cursor-pointer"
        title="View model library"
      >
        {#if downloadProgress}
          <!-- Compact download progress indicator -->
          <div class="relative w-4 h-4 flex-shrink-0">
            <svg class="w-4 h-4 -rotate-90" viewBox="0 0 20 20">
              <circle
                cx="10"
                cy="10"
                r="8"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                opacity="0.2"
              />
              <circle
                cx="10"
                cy="10"
                r="8"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-dasharray={2 * Math.PI * 8}
                stroke-dashoffset={2 *
                  Math.PI *
                  8 *
                  (1 - downloadProgress.percentage / 100)}
                class="text-blue-400 transition-all duration-300"
              />
            </svg>
            <div
              class="absolute inset-0 flex items-center justify-center text-[6px] font-mono text-blue-400"
            >
              {downloadProgress.count}
            </div>
          </div>
        {:else}
          <svg
            class="w-4 h-4"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          >
            <path d="M12 3v12" />
            <path d="M7 12l5 5 5-5" />
            <path d="M5 21h14" />
          </svg>
        {/if}
        <span class="hidden sm:inline">Library</span>
      </a>
      <a
        href="/#/server"
        class="text-xs md:text-sm text-white/70 hover:text-xeo-green transition-colors tracking-wider uppercase flex items-center gap-1.5 md:gap-2 cursor-pointer"
        title="Server connections for external tools"
      >
        <svg
          class="w-4 h-4"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <path
            d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"
          />
          <path
            d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"
          />
        </svg>
        <span class="hidden sm:inline">Server</span>
      </a>
      {#if showAdvanced}
        <a
          href="/#/advanced"
          class="text-xs md:text-sm text-white/70 hover:text-xeo-green transition-colors tracking-wider uppercase flex items-center gap-1.5 md:gap-2 cursor-pointer"
          title="Advanced cluster settings"
        >
          <svg
            class="w-4 h-4"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          >
            <circle cx="12" cy="12" r="3" />
            <path
              d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"
            />
          </svg>
          <span class="hidden sm:inline">Advanced</span>
        </a>
      {/if}
    </div>
    {#if clusterMemory}
      <div
        class="flex items-baseline gap-1.5 font-mono text-[10px] md:text-[11px] tracking-wider uppercase"
        title="Cluster memory available for models (available / total)"
        aria-label="Available VRAM {formatMemoryGb(
          clusterMemory.availableBytes,
        )} of {formatMemoryGb(clusterMemory.totalBytes)}"
      >
        <span class="text-white/40">Available VRAM</span>
        <span class="text-xeo-green"
          >{formatMemoryGb(clusterMemory.availableBytes)}</span
        >
        <span class="text-white/35"
          >/ {formatMemoryGb(clusterMemory.totalBytes)}</span
        >
      </div>
    {/if}
  </nav>
</header>
