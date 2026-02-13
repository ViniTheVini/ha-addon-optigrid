# OptiGrid - Home Assistant Add-on Repository

[![GitHub Release][releases-shield]][releases]
[![License][license-shield]](LICENSE)

Energy optimization add-on for Home Assistant with battery management and deferrable load scheduling.

## Installation

1. Click the button below to add this repository to your Home Assistant instance:

   [![Add Repository][add-repo-shield]][add-repo]

2. Or manually add the repository:
   - Navigate to **Settings** → **Add-ons** → **Add-on Store**
   - Click the **⋮** menu (top right) → **Repositories**
   - Add this URL: `https://github.com/ViniTheVini/ha-addon-optigrid`

3. Install the **OptiGrid** add-on from the store
4. Configure and start the add-on

## Add-ons in This Repository

### OptiGrid

Energy optimization using linear programming to optimize battery charging/discharging and schedule deferrable loads based on electricity prices and solar production.

**Features:**
- Battery charge/discharge optimization
- Deferrable load scheduling (EV charging, etc.)
- Solar PV integration
- Dynamic electricity pricing support
- Historical data analysis from Home Assistant
- REST API for automation integration

## Support

For issues and feature requests, please use the [GitHub issue tracker][issues].

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

---

[releases-shield]: https://img.shields.io/github/v/release/ViniTheVini/ha-addon-optigrid?style=flat-square
[releases]: https://github.com/ViniTheVini/ha-addon-optigrid/releases
[license-shield]: https://img.shields.io/github/license/ViniTheVini/ha-addon-optigrid?style=flat-square
[add-repo-shield]: https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg
[add-repo]: https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https%3A%2F%2Fgithub.com%2FViniTheVini%2Fha-addon-optigrid
[issues]: https://github.com/ViniTheVini/ha-addon-optigrid/issues