" Ported from ~/.config/nvim/lua/config/keymaps.lua
" Obsidian uses CodeMirror Vim, so this intentionally keeps only portable editor mappings.
" Enable Obsidian's Vim mode and vimrc loading for this file to take effect.

" Increment/decrement, select all, and centered half-page scrolling.
nmap + <C-a>
nmap - <C-x>
nmap <C-a> ggVG
nmap <C-d> <C-d>zz
nmap <C-u> <C-u>zz

" Preserve paste-over-selection without replacing the default yank register.
xmap p "_dP

" Clear Vim search highlight, mirroring <leader><leader> in Neovim.
nmap <Space><Space> :nohlsearch<CR>

" Insert a LaTeX align block, mirroring <leader>a in Neovim.
nmap <Space>a i$$\begin{align*}<CR><CR>\end{align*}$$<Esc>kA
